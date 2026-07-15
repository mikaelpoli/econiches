import warnings

import numpy as np
import pandas as pd
import yaml
from tabulate import tabulate

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed`"
)

from econiches.modeling.init import initialize
from econiches.modeling.run_pipeline import _export_metrics_csvs
from econiches.modeling.pipeline_backend import (optimize_threshold,
                                                 score,
                                                 plot_binary_confusion_heatmap)
from econiches.utils import choose_run_id, load_preds


def weighted_ensemble(
    results: dict[str, dict[str, np.ndarray]],
    field: str = "probs",
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Combine per-model arrays into a single weighted average.

    Stacks the chosen `field` (e.g. "probs" or "thresholds") from each
    model's results, then computes a weighted average across models.
    Weights are normalized to sum to 1, so relative magnitudes are all
    that matter. Equal weighting is used if none are given.

    Parameters
    ----------
    results:
        `{model_name: {"probs": ..., "thresholds": ..., ...}}` — one
        entry per model, each containing arrays of the same shape.
    field:
        Which key to pull from each model's results dict and combine.
    weights:
        `{model_name: weight}`. If `None`, all models are weighted
        equally.

    Returns
    -------
    Weighted average array, same shape as `results[any_model][field]`.
    """
    model_names = list(results.keys())

    stacked_values = np.stack(
        [results[name][field] for name in model_names],
        axis=0,
    )  # shape: (n_models, ...)

    if weights is None:
        raw_weights = np.ones(len(model_names))
    else:
        raw_weights = np.array([weights[name] for name in model_names])

    normalized_weights = raw_weights / raw_weights.sum()

    return np.tensordot(normalized_weights, stacked_values, axes=(0, 0))


if __name__ == "__main__":
    ctx = initialize(__file__)

    print("--- LOAD PREDICTIONS ---")

    annotation_types = ["cog", "ec", "pfam", "cazy", "ko"]
    indir = ctx.paths.root / "logs" / ctx.args.env

    file_suffix = input(
        "Prediction file suffix (Enter = full_test_preds.npz): "
    ).strip()

    if not file_suffix:
        file_suffix = "full_test_preds.npz"

    infiles    = {}
    results    = {}
    thresholds = {}
    rows       = []

    for ann in annotation_types:
        model   = input(f"\n{ann.upper()} model type: ").strip()
        ann_dir = indir / model / ann

        run_id  = choose_run_id(ann_dir)
        run_dir = ann_dir / run_id
        
        thresholds[ann] = run_dir / f"{run_id}_per_class_metrics.csv"
        infiles[ann]    = run_dir / f"{run_id}_{file_suffix}"
        results[ann]    = load_preds(infiles[ann])

        results[ann]["thresholds"] = (pd.read_csv(thresholds[ann],usecols=["model_split", "threshold"]))

        # Select threshold for desired model (types: train, test) (cv, full)
        results[ann]["thresholds"] = results[ann]["thresholds"].loc[results[ann]["thresholds"]["model_split"] == "val_cv", "threshold"]
        results[ann]["thresholds"] = results[ann]["thresholds"].squeeze().to_numpy()

        preds_shape      = results[ann]["preds"].shape
        probs_shape      = results[ann]["probs"].shape
        thresholds_shape = results[ann]["thresholds"].shape

        rows.append([
            ann,
            preds_shape,
            probs_shape,
            thresholds_shape
        ])

        ctx.log.info(f"{ann.upper()} model: {model.upper()}, run: {run_id}")

    table = tabulate(
        rows,
        headers=[
            "Annotation",
            "Preds shape",
            "Probs shape",
            "Thresholds shape"
            ],
        tablefmt="simple"
    )

    ctx.log.info(f"Model(s) prediction files:\n{table}")

    print("--- PREDICT ---")

    weights = ctx.config["model"]["weights"]
    ctx.config["model"]["weights"] = weights
    with open(ctx.paths.logs / "config.yaml", "w") as f:
        yaml.safe_dump(ctx.config, f)
    
    class_id_to_label = {i: l for i, l in enumerate(ctx.data["mlb"].classes_)}
    n_classes = len(class_id_to_label)

    ensemble_results = {}
    ensemble_results["probs"] = weighted_ensemble(results, "probs", weights)

    ensemble_results["thresholds"] = weighted_ensemble(results, "thresholds", weights)

    ensemble_results["preds"] = (ensemble_results["probs"] >= ensemble_results["thresholds"]).astype(int)
    ensemble_results["scores"] = score(
        ctx.data["y_test"],
        ensemble_results["probs"],
        class_names=ctx.class_names,
        thresholds=ensemble_results["thresholds"]
    )
    
    _export_metrics_csvs(
        ctx,
        {"test_full": ensemble_results["scores"]}
    )

    plot_binary_confusion_heatmap(
        ctx.data["y_test"],
        ensemble_results["preds"],
        ctx.class_names,
        thresholds=ensemble_results["thresholds"],
        save_path=ctx.paths.logs / f"{ctx.run_id}_test_confusion_heatmap.png"
    )
    
    exit(0)