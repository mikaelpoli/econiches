"""
Core ML utilities: per-class training, scoring, CV, grid search, and plotting.

Design notes
------------
* Every function that trains classifiers operates at the **per-class** level.
  The multi-label problem is decomposed into independent binary problems
  (one-vs-rest), which is common in pipelines where label sets are sparse
  and class frequencies vary wildly.

* "PU learning" (Positive-Unlabeled) is used because many MAGs lack confirmed
  annotations even when the pathway is likely present. Sample weights penalise
  unlabeled examples (weight = pu_beta < 1) while leaving confirmed positives
  at full weight.
"""

from __future__ import annotations

import itertools
import os
import pprint
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, precision_recall_curve,
                             precision_recall_fscore_support)

from econiches.modeling.init import transform_data

# ---------------------------------------------------------------------------
# Per-class training worker
# ---------------------------------------------------------------------------

def _train_worker(
    c: int,
    X_tr,
    X_va,
    y_tr: np.ndarray,
    y_va: np.ndarray | None,
    params: dict,
    *,
    pu_beta: float = 0.75,
) -> tuple[int, object | None, np.ndarray | None, np.ndarray | None]:
    """Fit a single binary classifier for class *c* using PU-weighted training.

    In Positive-Unlabeled (PU) learning, confirmed positives receive full
    weight while unlabeled examples (treated as putative negatives) are
    down-weighted by `pu_beta`.  This reflects the assumption that some
    unlabeled MAGs may carry the annotation but simply have not been confirmed.

    For Random Forests, weights are additionally scaled by inverse class
    frequency to counteract label imbalance before applying the PU discount.

    Parameters
    ----------
    c:
        Class (label) index to train.
    X_tr, X_va:
        Feature matrices for training and validation.  `X_va` may be
        `None` when called during full-data refit (no validation needed).
    y_tr, y_va:
        Multi-label binary target matrices.  Only column *c* is used here.
    params:
        Estimator hyperparameters.  Must include a `"model_type"` key
        (`"lr"` for SGDClassifier, `"rf"` for RandomForestClassifier).
    pu_beta:
        Down-weighting factor for unlabeled examples (0 < pu_beta ≤ 1).
        Lower values make the model more conservative about calling positives.
        For example, `pu_beta = 0.75` means that positive example weight
        1.5 times more than negative examples.

    Returns
    -------
    (c, clf, train_probs, val_probs)
        `clf` and both probability arrays are `None` when class *c* has
        no positive examples in the training split (untrainable).
    """
    params = dict(params)                     # avoid mutating the caller's dict
    model_type = params.pop("model_type")     # routing key, not a model kwarg

    y_c = y_tr[:, c]
    if y_c.sum() == 0:
        return c, None, None, None            # skip empty classes silently

    sample_weight = _pu_sample_weights(y_c, pu_beta)

    clf = _build_classifier(model_type, params)
    clf.fit(X_tr, y_c, sample_weight=sample_weight)

    train_probs = clf.predict_proba(X_tr)[:, 1]

    if X_va is None or y_va is None:
        return c, clf, train_probs, None

    val_probs = clf.predict_proba(X_va)[:, 1]
    return c, clf, train_probs, val_probs


def _pu_sample_weights(y_c: np.ndarray, pu_beta: float) -> np.ndarray:
    """Compute PU sample weights.

    Positives = 1.0, unlabeled = pu_beta.

    Weights are first scaled by inverse class frequency (to compensate for 
    label imbalance), then the PU discount is applied to unlabeled examples.
    This is preferred over `class_weight="balanced"` because it allows 
    the PU discount to operate on top of the frequency correction rather 
    than being overridden by it.
    """
    n, n_pos  = len(y_c), y_c.sum()
    n_unl     = n - n_pos
    pos_weight = (n / (2 * n_pos)) if n_pos > 0 else 1.0
    unl_weight = (n / (2 * n_unl)) if n_unl > 0 else 1.0

    return np.where(y_c == 1, pos_weight, unl_weight * pu_beta)


def _build_classifier(model_type: str, params: dict):
    """Instantiate the appropriate sklearn estimator."""
    if model_type == "lr":
        return SGDClassifier(**params)
    if model_type == "rf":
        return RandomForestClassifier(**params)
    raise ValueError(f"Unknown model_type '{model_type}'. Supported: 'lr', 'rf'.")


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------

def optimize_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    beta: float = 1.0,
) -> float:
    """Find the probability cut-off that maximises F-beta for one class.

    Rather than using a fixed 0.5 threshold, we search the full
    precision-recall curve and pick the threshold at which Fβ is highest.
    β = 1 (default) weights precision and recall equally.  Set β > 1 to
    prioritise recall (fewer missed positives), or β < 1 to prioritise
    precision (fewer false positives).

    Returns 0.5 when the precision-recall curve has no interior thresholds
    (degenerate class).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f_beta = (
        (1 + beta**2)
        * (precision * recall)
        / (beta**2 * precision + recall + 1e-9)
    )
    return float(thresholds[np.argmax(f_beta)]) if len(thresholds) else 0.5

# ---------------------------------------------------------------------------
# Multi-label scoring
# ---------------------------------------------------------------------------

def score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    thresholds: np.ndarray | None = None,
) -> dict:
    """Compute macro, micro, and per-class classification metrics.

    Classes with no positive examples in `y_true` are excluded from macro
    averages (they would trivially inflate or deflate the score).

    Parameters
    ----------
    y_true:
        Binary label matrix of shape (n_samples, n_classes).
    y_score:
        Predicted probability matrix of shape (n_samples, n_classes).
    class_names:
        Human-readable label for each class column.
    thresholds:
        Per-class probability cut-offs (from `optimize_threshold`).
        Defaults to 0.5 for all classes.

    Returns
    -------
    dict with keys:
        - `macro`     -> averaged metrics over classes with at least one positive
        - `micro`     -> metrics computed globally across all samples and classes
        - `per_class` -> list of per-class metric dicts
        - `coverage`  -> fraction of classes that had at least one positive
    """
    n_classes  = y_true.shape[1]
    thresholds = thresholds if thresholds is not None else np.full(n_classes, 0.5)

    per_class_records = []
    macro_accumulators: dict[str, list] = {
        "ap": [], "precision": [], "recall": [], "f1": [], "log_loss": [], "brier": []
    }

    for c in range(n_classes):
        y_c   = y_true[:, c]
        s_c   = y_score[:, c]
        support = int(y_c.sum())

        if support == 0:
            continue # class absent in this split: skip

        pred_c = (s_c >= thresholds[c]).astype(int)

        ap  = average_precision_score(y_c, s_c)
        p, r, f1, _ = precision_recall_fscore_support(
            y_c, pred_c, average="binary", zero_division=0
        )
        ll  = log_loss(y_c, np.clip(s_c, 1e-15, 1 - 1e-15))
        br  = brier_score_loss(y_c, s_c)

        for key, val in zip(["ap", "precision", "recall", "f1", "log_loss", "brier"],
                            [ap, p, r, f1, ll, br]):
            macro_accumulators[key].append(val)

        per_class_records.append({
            "class":     class_names[c],
            "support":   support,
            "ap":        ap,
            "f1":        f1,
            "precision": p,
            "recall":    r,
            "threshold": thresholds[c],
            "log_loss":  ll,
            "brier":     br,
        })

    def _mean(values: list) -> float:
        return float(np.mean(values)) if values else 0.0

    # Micro metrics are computed globally (all samples × all classes at once)
    pred_all = (y_score >= thresholds).astype(int)
    p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(
        y_true, pred_all, average="micro", zero_division=0
    )

    metrics = {
        "macro": {f"{k}_mean": _mean(v) for k, v in macro_accumulators.items()},
        "micro": {
            "ap":        float(average_precision_score(y_true, y_score, average="micro")),
            "f1":        float(f1_mi),
            "precision": float(p_mi),
            "recall":    float(r_mi),
            "log_loss":  float(log_loss(y_true.ravel(), np.clip(y_score.ravel(), 1e-15, 1 - 1e-15))),
            "brier":     float(brier_score_loss(y_true.ravel(), y_score.ravel())),
        },
        "per_class": per_class_records,
        "coverage":  len(per_class_records) / n_classes,
    }
    metrics["macro"]["ap_mdn"] = float(np.median(macro_accumulators["ap"]))
    metrics["macro"]["f1_mdn"] = float(np.median(macro_accumulators["f1"]))
    metrics["macro"]["f1_std"] = float(np.std(macro_accumulators["f1"]))
    return metrics


def macro_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean per-class log loss, skipping classes without both label values.

    Standard `sklearn.metrics.log_loss` computed globally can be dominated
    by abundant classes.  This macro version weights every class equally,
    which is more informative when class frequencies vary across orders of
    magnitude (as is typical in functional annotation datasets).
    """
    per_class_losses = [
        log_loss(y_true[:, c], y_prob[:, c], labels=[0, 1])
        for c in range(y_true.shape[1])
        if len(np.unique(y_true[:, c])) == 2    # skip constant-label classes
    ]
    return float(np.mean(per_class_losses)) if per_class_losses else np.nan


# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------

def predict_ensemble(fold_models: dict, X) -> np.ndarray:
    """Average predicted probabilities across all CV fold models.

    Each fold produced one fitted classifier per class.  Averaging their
    predictions reduces variance compared to any single fold model — a
    standard ensembling technique.

    Parameters
    ----------
    fold_models:
        `{class_index: [clf_fold1, clf_fold2, ...]}` as built by
        `cross_validation`.
    X:
        Feature matrix for the samples to predict (typically the test set).

    Returns
    -------
    Array of shape (n_samples, n_classes) where entry [i, c] is the mean
    ensemble probability that sample i belongs to class c.
    """
    n_classes = max(fold_models) + 1
    probs     = np.zeros((X.shape[0], n_classes), dtype=np.float32)

    for c, models in fold_models.items():
        if models:
            probs[:, c] = np.mean(
                [clf.predict_proba(X)[:, 1] for clf in models], axis=0
            )
    return probs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_binary_confusion_heatmap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    thresholds: np.ndarray | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot a per-class normalised confusion heatmap (TP / FP / FN rates).

    Each column corresponds to one annotation class (e.g. a KEGG Ortholog or
    CAZyme family).  Rows show what fraction of the per-class predictions fall
    into true positives, false positives, and false negatives — normalised by
    TP + FP + FN so that columns are comparable regardless of class size.

    Parameters
    ----------
    y_true:
        Binary label matrix (n_samples, n_classes).
    y_score:
        Predicted probabilities (n_samples, n_classes).
    class_names:
        Label for each class column.
    thresholds:
        Per-class cut-offs.  Defaults to 0.5.
    save_path:
        File path to save the figure.  Displays interactively if `None`.
    """
    n_classes  = len(class_names)
    thresholds = np.full(n_classes, 0.5) if thresholds is None else np.asarray(thresholds)

    y_hat = (y_score >= thresholds).astype(int)
    tp    = (y_true       * y_hat).sum(axis=0)
    fp    = ((1 - y_true) * y_hat).sum(axis=0)
    fn    = (y_true       * (1 - y_hat)).sum(axis=0)
    class_names = np.array(class_names)
    denom = tp + fp + fn + 1e-9

    # Matrix rows: TP%, FP%, FN% — each value ∈ [0, 1]
    rate_matrix = np.vstack([tp / denom, fp / denom, fn / denom])
    
    # sort classes by TP (desc), FP (asc), FN (asc)
    tp_r = rate_matrix[0]
    fp_r = rate_matrix[1]
    fn_r = rate_matrix[2]

    sort_idx = np.lexsort((fn_r, fp_r, -tp_r))
    
    rate_matrix = rate_matrix[:, sort_idx]
    class_names = np.array(class_names)[sort_idx]

    fig, ax = plt.subplots(figsize=(max(12, n_classes * 0.5), 6))
    sns.heatmap(
        rate_matrix, annot=True, fmt=".2f", cmap="viridis",
        xticklabels=class_names, yticklabels=["TP rate", "FP rate", "FN rate"],
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Per-class Normalised Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_learning_curve(
    train_metric: list[float],
    val_metric:   list[float],
    *,
    metric_name: str  = "Macro AP",
    save_path:   Path | None = None,
    show:        bool = False,
) -> None:
    """Plot train and validation metric histories across CV folds.

    A widening gap between the train and validation curves suggests
    overfitting; curves that both plateau low suggest underfitting or
    insufficient data for the chosen model complexity.

    Parameters
    ----------
    train_metric, val_metric:
        Per-fold metric values (e.g. from `cv_results["ap_train_hist"]`).
    metric_name:
        Y-axis label and plot title.
    save_path:
        Save the figure here (PNG recommended).  Skipped if `None`.
    show:
        Display the figure interactively in addition to saving.
    """
    folds = np.arange(1, len(train_metric) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(folds, train_metric, marker="o", color="steelblue", label="Train")
    ax.plot(folds, val_metric,   marker="o", color="darkorange", label="Validation")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Cross-Validation Learning Curve — {metric_name}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validation(
    X,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: list[str],
    params: dict,
    log,
    ctx,
) -> dict:
    """Group-stratified multi-label cross-validation.

    MAGs (metagenome-assembled genomes) from the same metagenome sample are
    kept together in either the training or validation split — never split
    across both.  This prevents data leakage that would arise from the
    ecological correlation between MAGs co-assembled from the same sample.

    Stratification ensures that each label's prevalence is roughly equal
    across folds despite the group constraint, using `MultilabelStratifiedKFold`.

    Parameters
    ----------
    X:
        Feature matrix (n_samples, n_features); may be dense or CSR sparse.
    y:
        Binary label matrix (n_samples, n_classes).
    groups:
        Per-sample metagenome identifier; MAGs sharing a group stay together.
    class_names:
        Human-readable annotation label for each class column.
    params:
        Hyperparameters forwarded to `_train_worker`.
    log:
        Run logger.
    ctx:
        Run context (used for scaler config and output paths).

    Returns
    -------
    dict with keys:
        - `in_fold_probs`     -> average train probabilities across folds (n_samples, n_classes)
        - `oof_probs`         -> out-of-fold validation probabilities (n_samples, n_classes)
        - `fold_models`       -> {class_index: [clf_fold1, clf_fold2, ...]}
        - `ap_train_hist`     -> per-fold macro AP on training split
        - `ap_val_hist`       -> per-fold macro AP on validation split
        - `log_loss_tr_hist`  -> per-fold macro log loss on training split
        - `log_loss_val_hist` -> per-fold macro log loss on validation split
    """
    n_classes = y.shape[1]
    n_splits  = 3

    # --- Build group-level label matrix for stratified splitting ---
    # We stratify at the *group* (metagenome) level, not the sample level,
    # to respect the group constraint while balancing label prevalence.
    unique_groups, group_targets = _group_label_matrix(groups, y, n_classes)

    mskf       = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    groups_arr = np.asarray(groups)

    # Accumulators
    in_fold_probs_sum = np.zeros_like(y, dtype=float)
    oof_probs         = np.zeros_like(y, dtype=float)
    fold_models: dict = defaultdict(list)
    ap_train_hist, ap_val_hist          = [], []
    log_loss_tr_hist, log_loss_val_hist = [], []
    fold_composition_records            = []

    for fold, (train_group_idx, val_group_idx) in enumerate(
        mskf.split(unique_groups, group_targets)
    ):
        log.info("--- Fold %d ---", fold + 1)

        train_groups = set(unique_groups[train_group_idx])
        val_groups   = set(unique_groups[val_group_idx])

        # Safety check: confirm no metagenome appears in both splits
        if overlap := train_groups & val_groups:
            raise ValueError(
                f"DATA LEAK detected in fold {fold + 1}: "
                f"{len(overlap)} metagenome group(s) appear in both train and val."
            )
        log.info("-> Leakage check: PASSED")

        tr_idx  = np.where(np.isin(groups_arr, list(train_groups)))[0]
        val_idx = np.where(np.isin(groups_arr, list(val_groups)))[0]

        y_tr = y[tr_idx]
        y_va = y[val_idx]

        X_tr_scaled, X_va_scaled, scaler = transform_data(
            X[tr_idx], X[val_idx], log,
            scaler=ctx.config["scaler"], log_scale=False,
        )

        # --- Label distribution audit (saved to CSV after all folds) ---
        fold_composition_records.extend(
            _fold_composition_records(fold, tr_idx, val_idx, y_tr, y_va, class_names, n_classes)
        )

        log.info("-> Training %d binary classifiers in parallel...", n_classes)
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_train_worker)(
                c=c,
                X_tr=X_tr_scaled,
                X_va=X_va_scaled,
                y_tr=np.ascontiguousarray(y_tr),
                y_va=np.ascontiguousarray(y_va),
                params=params,
            )
            for c in range(n_classes)
        )

        fold_tr_probs  = np.zeros_like(y_tr, dtype=float)
        fold_val_probs = np.zeros_like(y_va, dtype=float)

        for c, clf, tr_p, va_p in results:
            if clf is not None:
                fold_models[c].append(clf)
                fold_tr_probs[:, c]  = tr_p
                fold_val_probs[:, c] = va_p

        in_fold_probs_sum[tr_idx] += fold_tr_probs
        oof_probs[val_idx]         = fold_val_probs

        # Metrics only over classes active in both splits
        active = np.where((y_tr.sum(0) > 0) & (y_va.sum(0) > 0))[0]
        ap_tr  = average_precision_score(y_tr[:, active], fold_tr_probs[:, active],  average="macro") if len(active) else 0.0
        ap_val = average_precision_score(y_va[:, active], fold_val_probs[:, active], average="macro") if len(active) else 0.0
        ll_tr  = macro_log_loss(y_tr[:, active], fold_tr_probs[:, active])
        ll_val = macro_log_loss(y_va[:, active], fold_val_probs[:, active])

        ap_train_hist.append(ap_tr);   ap_val_hist.append(ap_val)
        log_loss_tr_hist.append(ll_tr); log_loss_val_hist.append(ll_val)

        log.info(
            "Fold %d | Train AP: %.4f | Val AP: %.4f | Train LogLoss: %.4f | Val LogLoss: %.4f",
            fold + 1, ap_tr, ap_val, ll_tr, ll_val,
        )

    # Save per-fold label distribution audit
    composition_path = ctx.paths.logs / "cv_label_compositions.csv"
    pd.DataFrame(fold_composition_records).to_csv(composition_path, index=False)
    log.info("-> Saved fold label distributions to: %s", composition_path)

    # Average in-fold probs over the (n_splits - 1) folds each sample appeared in
    in_fold_probs = in_fold_probs_sum / (n_splits - 1)

    return {
        "in_fold_probs":      in_fold_probs,
        "oof_probs":          oof_probs,
        "fold_models":        fold_models,
        "ap_train_hist":      ap_train_hist,
        "ap_val_hist":        ap_val_hist,
        "log_loss_tr_hist":   log_loss_tr_hist,
        "log_loss_val_hist":  log_loss_val_hist,
    }


def _group_label_matrix(
    groups: np.ndarray,
    y: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a group-level presence/absence matrix for stratified splitting.
 
    A label is considered *present* for a group if at least one member MAG
    carries it (max pooling).  This lets ``MultilabelStratifiedKFold`` balance
    label frequencies across folds even when working at the group level.
    """
    df = pd.DataFrame({"group": groups})
    for c in range(n_classes):
        df[f"l{c}"] = y[:, c]
 
    unique_groups = df["group"].unique()
    group_targets = df.groupby("group").max().loc[unique_groups].values
    return unique_groups, group_targets


def _fold_composition_records(
    fold: int,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    class_names: list[str],
    n_classes: int,
) -> list[dict]:
    """Build per-class label distribution records for one fold.

    These are written to CSV after all folds complete, providing an audit
    trail that confirms the CV split preserved label balance.
    """
    records = []
    for c in range(n_classes):
        tr_count  = int(y_tr[:, c].sum())
        val_count = int(y_va[:, c].sum())
        total     = tr_count + val_count

        records.append({
            "fold":                    fold + 1,
            "class_index":             c,
            "class_name":              class_names[c] if class_names else f"Class_{c}",
            "train_positive_count":    tr_count,
            "train_total_samples":     len(tr_idx),
            "train_positive_pct":      round(tr_count  / total * 100, 2) if total else 0.0,
            "val_positive_count":      val_count,
            "val_total_samples":       len(val_idx),
            "val_positive_pct":        round(val_count / total * 100, 2) if total else 0.0,
        })
    return records


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(
    X,
    y: np.ndarray,
    groups: np.ndarray,
    param_grid: dict,
    fixed_params: dict,
    log,
    ctx,
    *,
    parallel_combos: bool = False,
    n_jobs_combos: int = 2,
    best_params_file: Path | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Exhaustive hyperparameter search using cross-validated OOF performance.

    Each candidate combination is evaluated with a full group-stratified CV
    (see `_cv_for_params`).  The winning combination maximises macro average
    precision, with macro F1 and micro F1 used as tie-breakers.

    Parameters
    ----------
    param_grid:
        `{param_name: [value1, value2, ...]}` — all combinations are tried.
    fixed_params:
        Parameters that do not vary (merged with every candidate combination).
    parallel_combos:
        Run combinations in parallel.  Caution: each combination already
        parallelises internally, so this multiplies CPU usage sharply.
    n_jobs_combos:
        Number of parallel combination workers (only used when
        `parallel_combos=True`).
    best_params_file:
        If given, save the winning parameters to this YAML file.

    Returns
    -------
    (best_params, results_df)
        `best_params` is ready to pass to `_train_worker`.
        `results_df` contains all combinations ranked by OOF performance.
    """
    combos   = _expand_param_grid(param_grid)
    n_combos = len(combos)
    log.info("Grid search: %d hyperparameter combinations to evaluate.", n_combos)

    def _evaluate(combo_idx: int, searched_params: dict) -> dict:
        params = {**fixed_params, **searched_params}
        ap, f1_macro, f1_micro = _cv_for_params(
            params, X, y, groups, log, ctx, combo_idx, n_combos
        )
        return {**searched_params, "oof_macro_ap": ap, "oof_macro_f1": f1_macro, "oof_micro_f1": f1_micro}

    if parallel_combos:
        log.info(
            "Running combinations in parallel (n_jobs=%d)."
            "Note: each combination also parallelises internally — keep n_jobs_combos low.",
            n_jobs_combos,
        )
        records = Parallel(n_jobs=n_jobs_combos, backend="loky")(
            delayed(_evaluate)(i, combo) for i, combo in enumerate(combos)
        )
    else:
        records = [_evaluate(i, combo) for i, combo in enumerate(combos)]

    results_df = (
        pd.DataFrame(records)
        .sort_values(["oof_macro_ap", "oof_macro_f1", "oof_micro_f1"], ascending=False)
        .reset_index(drop=True)
    )

    best_searched = {k: results_df.iloc[0][k] for k in param_grid}
    best_params   = {**fixed_params, **best_searched}

    log.info(
        "Grid search complete.\n"
        "  Best OOF AP       : %.4f\n"
        "  Best OOF macro F1 : %.4f\n"
        "  Best OOF micro F1 : %.4f\n"
        "  Best params       :\n%s",
        results_df.iloc[0]["oof_macro_ap"],
        results_df.iloc[0]["oof_macro_f1"],
        results_df.iloc[0]["oof_micro_f1"],
        pprint.pformat(best_params),
    )

    if best_params_file:
        _save_yaml(best_params, best_params_file)
        log.info("Saved best params to %s", best_params_file)

    return best_params, results_df


def _cv_for_params(
    params: dict,
    X,
    y: np.ndarray,
    groups: np.ndarray,
    log,
    ctx,
    combo_idx: int,
    n_combos: int,
) -> tuple[float, float, float]:
    """Run a lightweight CV to score one hyperparameter combination.

    Unlike the main `cross_validation` function, this version stores no
    fold models and collects only OOF probabilities for the final score.
    It is intentionally minimal to keep grid search fast.

    Returns
    -------
    (oof_macro_ap, oof_macro_f1, oof_micro_f1)
    """
    n_classes = y.shape[1]
    n_splits  = 3

    unique_groups, group_targets = _group_label_matrix(groups, y, n_classes)
    mskf       = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=420)
    groups_arr = np.asarray(groups)
    oof_probs  = np.zeros_like(y, dtype=float)

    for fold, (train_group_idx, val_group_idx) in enumerate(
        mskf.split(unique_groups, group_targets)
    ):
        log.info("Combo %d/%d | Fold %d/%d", combo_idx + 1, n_combos, fold + 1, n_splits)

        tr_idx  = np.where(np.isin(groups_arr, list(unique_groups[train_group_idx])))[0]
        val_idx = np.where(np.isin(groups_arr, list(unique_groups[val_group_idx])))[0]

        y_tr = y[tr_idx]
        y_va = y[val_idx]

        X_tr_scaled, X_va_scaled, scaler = transform_data(
            X[tr_idx], X[val_idx], log,
            scaler=ctx.config["scaler"], log_scale=False,
        )

        fold_val_probs = np.zeros_like(y_va, dtype=float)

        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_train_worker)(
                c=c,
                X_tr=X_tr_scaled,
                X_va=X_va_scaled,
                y_tr=np.ascontiguousarray(y_tr),
                y_va=np.ascontiguousarray(y_va),
                params=params,
            )
            for c in range(n_classes)
        )

        for c, clf, _, va_p in results:
            if clf is not None:
                fold_val_probs[:, c] = va_p

        oof_probs[val_idx] = fold_val_probs

    # Evaluate only on classes that have at least one positive across all samples
    active = np.where(y.sum(0) > 0)[0]
    oof_at_threshold = (oof_probs[:, active] >= 0.5).astype(int)

    oof_ap = average_precision_score(y[:, active], oof_probs[:, active], average="macro")
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y[:, active], oof_at_threshold, average="macro", zero_division=0
    )
    _, _, f1_micro, _ = precision_recall_fscore_support(
        y[:, active], oof_at_threshold, average="micro", zero_division=0
    )

    return float(oof_ap), float(f1_macro), float(f1_micro)


def _expand_param_grid(param_grid: dict) -> list[dict]:
    """Convert a grid dict into a flat list of all hyperparameter combinations.

    Example
    -------
    >>> _expand_param_grid({"C": [0.01, 0.1], "l1_ratio": [0.3, 0.7]})
    [{"C": 0.01, "l1_ratio": 0.3},
     {"C": 0.01, "l1_ratio": 0.7},
     {"C": 0.1,  "l1_ratio": 0.3},
     {"C": 0.1,  "l1_ratio": 0.7}]
    """
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _save_yaml(data: dict, path: Path) -> None:
    """Serialise `data` to YAML, converting numpy scalars to native Python types."""
    def _to_python(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        return obj

    with open(path, "w") as f:
        yaml.safe_dump(_to_python(data), f)