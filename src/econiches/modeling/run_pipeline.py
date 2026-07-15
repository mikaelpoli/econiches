"""
Core experiment runner: CV → full refit → test evaluation → export.
"""

from __future__ import annotations

import os

os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"]      = "1"

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from scipy.sparse import csr_matrix

from econiches.modeling import pipeline_backend as pb

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiment(ctx: SimpleNamespace, config: dict, *, sparse: bool = True) -> dict:
    """Run the full ML experiment pipeline for a single configuration.

    Stages
    ------
    1. Prepare feature arrays (NaN fill, dtype cast, optional sparse conversion).
    2. Hyperparameter search (grid search) or fall back to config defaults.
    3. Cross-validation → OOF probabilities → per-class threshold optimisation.
    4. Full-data refit with optional feature scaling.
    5. Test-set evaluation for both CV-ensemble and full-refit models.
    6. Export metrics (macro / micro / per-class CSVs), learning curves,
       confusion heatmaps, compressed prediction arrays, and feature importances.

    Parameters
    ----------
    ctx:
        Run context produced by `initialize()`.
    config:
        Experiment config dict.  Must contain at least one of
        `PARAM_GRID` (triggers grid search) or
        `CV_MODEL_PARAMS` / `FULL_MODEL_PARAMS`.
    sparse:
        When `True` (default), convert feature matrices to CSR sparse format
        before training.  Disable for dense-only estimators.

    Returns
    -------
    dict with keys `train`, `val`, `test`, `cv_models`, `full_models`.
    """
    log         = ctx.log
    class_names = ctx.class_names
    n_classes   = len(class_names)

    # ------------------------------------------------------------------ #
    # 1. Prepare arrays                                                  #
    # ------------------------------------------------------------------ #
    X_tr_raw, X_te_raw, y_tr, y_test, groups = _prepare_arrays(ctx)
    X_tr, X_te = _to_sparse_or_dense(X_tr_raw, X_te_raw, sparse, log)

    # ------------------------------------------------------------------ #
    # 2. Hyperparameter selection                                        #
    # ------------------------------------------------------------------ #
    cv_params, full_params = _select_params(ctx, config, X_tr, y_tr, groups, log)

    # ------------------------------------------------------------------ #
    # 3. Cross-validation + threshold optimisation                       #
    # ------------------------------------------------------------------ #
    log.info("\n=== Cross-Validation ===")
    cv_results = pb.cross_validation(X_tr, y_tr, groups, class_names, cv_params, log, ctx)

    thresholds = np.array([
        pb.optimize_threshold(y_tr[:, c], cv_results["oof_probs"][:, c])
        for c in range(n_classes)
    ])

    cv_tr_metrics  = pb.score(y_tr, cv_results["in_fold_probs"], class_names, thresholds)
    cv_val_metrics = pb.score(y_tr, cv_results["oof_probs"],     class_names, thresholds)

    _save_learning_curves(ctx, cv_results)

    # ------------------------------------------------------------------ #
    # 4. Full-data refit                                                 #
    # ------------------------------------------------------------------ #
    log.info("\n=== Full-data refit ===")
    X_tr_scaled, X_te_scaled, scaler = pb.transform_data(
        X_tr, X_te, log, scaler=ctx.config["scaler"], log_scale=False
    )
    dump(scaler, ctx.paths.logs / "scaler.joblib")

    full_models, full_tr_probs = _fit_full_models(ctx, X_tr_scaled, y_tr, full_params, n_classes)
    full_tr_metrics = pb.score(y_tr, full_tr_probs, class_names, thresholds)

    # ------------------------------------------------------------------ #
    # 5. Test evaluation                                                 #
    # ------------------------------------------------------------------ #
    log.info("\n=== Test evaluation ===")
    cv_test_probs   = pb.predict_ensemble(cv_results["fold_models"], X_te_scaled)
    full_test_probs = _predict_full(full_models, X_te_scaled, X_te.shape[0], n_classes)

    test_cv_metrics   = pb.score(y_test, cv_test_probs,   class_names, thresholds)
    test_full_metrics = pb.score(y_test, full_test_probs, class_names, thresholds)

    # ------------------------------------------------------------------ #
    # 6. Export                                                          #
    # ------------------------------------------------------------------ #
    all_metrics = {
        "train_cv":   cv_tr_metrics,
        "val_cv":     cv_val_metrics,
        "train_full": full_tr_metrics,
        "test_cv":    test_cv_metrics,
        "test_full":  test_full_metrics,
    }
    _export_metrics_csvs(ctx, all_metrics)
    _export_confusion_heatmaps(ctx, all_metrics, y_tr, y_test, cv_results, full_tr_probs, cv_test_probs, full_test_probs, thresholds, class_names)
    _export_test_predictions(ctx, cv_test_probs, full_test_probs, thresholds)
    _log_metric_summary(log, cv_tr_metrics, cv_val_metrics, full_tr_metrics, test_cv_metrics, test_full_metrics)

    return {
        "train":      {"cv": cv_tr_metrics,   "full": full_tr_metrics},
        "val":        cv_val_metrics,
        "test":       {"cv": test_cv_metrics,  "full": test_full_metrics},
        "cv_models":  cv_results["fold_models"],
        "full_models": full_models,
    }


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _prepare_arrays(ctx: SimpleNamespace):
    """Extract and clean raw arrays from the run context."""
    data = ctx.data

    def to_array(x):
        return x.values if isinstance(x, pd.DataFrame) else x

    def to_1d(x):
        return x.values if isinstance(x, pd.Series) else x

    X_tr_raw = np.nan_to_num(to_array(data["X_train"]).astype(np.float32))
    X_te_raw = np.nan_to_num(to_array(data["X_test"]).astype(np.float32))
    y_tr     = to_array(data["y_train"])
    y_test   = to_array(data["y_test"])
    groups   = to_1d(data.get("groups_train"))

    return X_tr_raw, X_te_raw, y_tr, y_test, groups


def _to_sparse_or_dense(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    sparse: bool,
    log,
) -> tuple:
    """Cast arrays to float64 and optionally convert to CSR sparse format."""
    if sparse:
        log.info("--- Converting to sparse format ---")
        return csr_matrix(X_tr.astype(np.float64)), csr_matrix(X_te.astype(np.float64))
    else:
        log.info("--- Skipping sparse conversion ---")
        return X_tr.astype(np.float64), X_te.astype(np.float64)


def _select_params(ctx, config, X_tr, y_tr, groups, log) -> tuple[dict, dict]:
    """Return (cv_params, full_params), running grid search when configured."""
    if config.get("PARAM_GRID"):
        log.info("=== Launching Grid Search ===")
        best_params, _ = pb.grid_search(
            X                = X_tr,
            y                = y_tr,
            groups           = groups,
            param_grid       = config["PARAM_GRID"],
            fixed_params     = config["FIXED_MODEL_PARAMS"],
            log              = log,
            ctx              = ctx,
            parallel_combos  = False,
            best_params_file = ctx.paths.logs / f"{ctx.run_id}_best_params.yaml",
        )
        # Grid search yields a single best config used for both CV and full refit
        return best_params, best_params
    else:
        log.info("Skipping Grid Search; using config parameters.")
        return config["CV_MODEL_PARAMS"], config["FULL_MODEL_PARAMS"]


def _save_learning_curves(ctx, cv_results: dict) -> None:
    """Plot and save AP and log-loss learning curves from CV history."""
    curves = {
        "Macro AP": {
            "train": cv_results["ap_train_hist"],
            "val":   cv_results["ap_val_hist"],
            "path":  ctx.paths.plots / f"{ctx.run_id}_cv_ap.png",
        },
        "Macro Log Loss": {
            "train": cv_results["log_loss_tr_hist"],
            "val":   cv_results["log_loss_val_hist"],
            "path":  ctx.paths.plots / f"{ctx.run_id}_cv_ll.png",
        },
    }
    for metric_name, curve in curves.items():
        pb.plot_learning_curve(
            curve["train"], curve["val"],
            metric_name=metric_name,
            save_path=curve["path"],
            show=False,
        )
        ctx.log.info("Saved %s learning curve to %s", metric_name, curve["path"])


def _fit_full_models(
    ctx: SimpleNamespace,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    params: dict,
    n_classes: int,
) -> tuple[dict, np.ndarray]:
    """Refit one classifier per class on the full training set in parallel.

    Also extracts and saves per-class feature importances (non-zero only).

    Returns
    -------
    full_models : {class_index: fitted_classifier}
    full_probs  : (n_train, n_classes) probability array
    """
    importance_dir = ctx.paths.logs / "feature_importance"
    importance_dir.mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(pb._train_worker)(
            c=c, X_tr=X_tr, X_va=None, y_tr=y_tr, y_va=None, params=params
        )
        for c in range(n_classes)
    )

    full_models: dict      = {}
    full_probs             = np.zeros_like(y_tr, dtype=float)

    for c, clf, probs, _ in results:
        if clf is None:
            continue

        full_models[c]   = clf
        full_probs[:, c] = probs

        _save_feature_importance(ctx, clf, c, importance_dir)

    ctx.log.info("Saved feature importance results for all classes to %s", importance_dir)
    return full_models, full_probs


def _save_feature_importance(ctx, clf, class_index: int, importance_dir: Path) -> None:
    """Extract, filter, and persist feature importances for one class."""
    class_name = ctx.id_to_lab[class_index].lower()

    coef = clf.coef_.ravel() if hasattr(clf, "coef_") else clf.feature_importances_

    importance = (
        pd.Series(coef, index=ctx.data["feature_names_train"])
        .sort_values(key=np.abs, ascending=False)
    )
    # Drop coefficients indistinguishable from zero (floating-point noise)
    importance = importance[importance.abs() > 1e-12]
    importance.to_csv(importance_dir / f"{class_name}.csv")

    ctx.log.info(
        "Found %d relevant features for class '%s'", len(importance), class_name
    )


def _predict_full(
    full_models: dict,
    X_te: np.ndarray,
    n_samples: int,
    n_classes: int,
) -> np.ndarray:
    """Collect per-class probabilities from the full-refit models."""
    probs = np.zeros((n_samples, n_classes), dtype=float)
    for c, clf in full_models.items():
        probs[:, c] = clf.predict_proba(X_te)[:, 1]
    return probs


def _export_metrics_csvs(ctx: SimpleNamespace, all_metrics: dict) -> None:
    """Write macro, micro, and per-class metric tables to CSV files."""
    def _records(level: str) -> list[dict]:
        return [{"model_split": split, **metrics[level]} for split, metrics in all_metrics.items()]

    for level, suffix in [("macro", "macro"), ("micro", "micro")]:
        path = ctx.paths.logs / f"{ctx.run_id}_{suffix}_metrics.csv"
        pd.DataFrame(_records(level)).to_csv(path, index=False)
        ctx.log.info("Saved %s metrics: %s", suffix.capitalize(), path)

    per_class_records = [
        {"model_split": split, **item}
        for split, metrics in all_metrics.items()
        for item in metrics["per_class"]
    ]
    per_class_path = ctx.paths.logs / f"{ctx.run_id}_per_class_metrics.csv"
    pd.DataFrame(per_class_records).to_csv(per_class_path, index=False)
    ctx.log.info("Saved per-class metrics: %s", per_class_path)


def _export_confusion_heatmaps(
    ctx, all_metrics, y_tr, y_test, cv_results, full_tr_probs, cv_test_probs, full_test_probs, thresholds, class_names
) -> None:
    """Save binary confusion heatmaps for every split x model combination."""
    # Map each split name to the ground-truth labels and the two score arrays
    split_data = {
        "train_cv":   (y_tr,    cv_results["in_fold_probs"], full_tr_probs),
        "val_cv":     (y_tr,    cv_results["oof_probs"],     full_tr_probs),
        "test_cv":    (y_test,  cv_test_probs,               full_test_probs),
        "train_full": (y_tr,    cv_results["in_fold_probs"], full_tr_probs),
        "test_full":  (y_test,  cv_test_probs,               full_test_probs),
    }
    for split_name in all_metrics:
        y_true, cv_scores, full_scores = split_data[split_name]
        model_type = split_name.split("_")[1]   # "cv" or "full"
        y_score    = cv_scores if model_type == "cv" else full_scores

        save_path = ctx.paths.plots / f"{ctx.run_id}_{split_name}_confusion.png"
        pb.plot_binary_confusion_heatmap(
            y_true, y_score,
            class_names=class_names,
            thresholds=thresholds,
            save_path=save_path,
        )
        ctx.log.info("Saved confusion heatmap for %s to %s", split_name, save_path)


def _export_test_predictions(
    ctx: SimpleNamespace,
    cv_test_probs: np.ndarray,
    full_test_probs: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    """Save binarised predictions and raw probabilities for both test models."""
    print("\n=== Saving boolean predictions for all models ===")

    for model_name, probs in [("cv", cv_test_probs), ("full", full_test_probs)]:
        path = ctx.paths.logs / f"{ctx.run_id}_{model_name}_test_preds.npz"
        np.savez_compressed(
            path,
            preds=(probs >= thresholds).astype(int),
            probs=probs.astype(np.float32),
        )
        ctx.log.info("Saved %s test predictions to %s", model_name, path)


def _log_metric_summary(log, cv_tr, cv_val, full_tr, test_cv, test_full) -> None:
    """Log a compact performance table for quick visual inspection."""
    def _row(label: str, m: dict) -> str:
        mac, mic = m["macro"], m["micro"]
        return (
            f"{label:<6}"
            f"  AP   macro(mdn)={mac['ap_mdn']:.4f}  micro={mic['ap']:.4f}"
            f"  F1   macro(mdn)={mac['f1_mdn']:.4f}  micro={mic['f1']:.4f}"
            f"  LogLoss={mac['log_loss_mean']:.4f}"
        )

    log.info(
        "\nTRAIN METRICS\n%s\n%s"
        "\nVAL METRICS (CV only)\n%s"
        "\nTEST METRICS\n%s\n%s",
        _row("CV",   cv_tr),
        _row("FULL", full_tr),
        _row("CV",   cv_val),
        _row("CV",   test_cv),
        _row("FULL", test_full),
    )