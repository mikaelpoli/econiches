"""
Evaluate the final trained model (or weighted ensemble across feature sets)
on a completely new, independent validation set.

CRITICAL PRINCIPLE: this script must not fit anything. Every transformer
(scaler/vectorizer/encoder) and every model must be *loaded* from what was
already fit during training. The only thing this script does with the new
data is `.transform()` and `.predict_proba()`.

Thresholds are loaded from each model's val_cv row in its per-class metrics
CSV — the CV/OOF-derived threshold, never touched by the training test set
or by this new validation set. This is the same threshold your main
pipeline already uses for test_full scoring, so results here are directly
comparable to your reported test_full metrics.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from econiches.modeling.pipeline_backend import score, plot_binary_confusion_heatmap
from econiches.modeling import preprocessing

from sklearn.preprocessing import MaxAbsScaler


def export_metrics_csvs(
    all_metrics: dict,
    logs_dir: str | Path,
) -> None:
    """Write macro, micro, and per-class metric tables to CSV files."""

    logs_dir = Path(logs_dir)

    def _records(level: str) -> list[dict]:
        return [
            {"model_split": split, **metrics[level]}
            for split, metrics in all_metrics.items()
        ]
    
    # macro and micro
    for level, suffix in [("macro", "macro"), ("micro", "micro")]:
        path = logs_dir / f"{suffix}_metrics.csv"
        pd.DataFrame(_records(level)).to_csv(path, index=False)

    # per class
    per_class_records = [
        {"model_split": split, **item}
        for split, metrics in all_metrics.items()
        for item in metrics["per_class"]
    ]

    per_class_path = logs_dir / f"per_class_metrics.csv"
    pd.DataFrame(per_class_records).to_csv(per_class_path, index=False)


def select_features(df, fam, extra_col=None):
    cols = [col for col in df.columns if col.startswith(fam)]

    if extra_col is not None:
        cols.append(extra_col)

    return df[cols]

    
if __name__ == "__main__":
    X_val = pd.read_parquet("./validation/data/filtered/full/features.parquet")
    y_val = pd.read_csv("./validation/data/filtered/full/labels.csv")

    
    X_val = select_features(X_val, "KO", "mag_id")
    X_val.fillna(0, inplace=True)
    print(f"X (DataFrame) shape: {X_val.shape}")

    X, y, groups, ids, mlb, feature_names = preprocessing.prepare_multilabel_data(X_val, y_val, "environment")

    print(f"X (ndarray) shape: {X.shape}")
    print(f"y (ndarray) shape: {y.shape}")
    print(f"Classes: {mlb.classes_}")

    scaler = joblib.load("./logs/environment/lr/ko/20260625_111354/scaler.joblib")
    X_scaled = scaler.transform(X)
    print(f"X (scaled) shape: {X_scaled.shape}")

    models = joblib.load("./logs/environment/lr/ko/20260625_111354/model.joblib")
    thresholds = pd.read_csv("./logs/environment/lr/ko/20260625_111354/20260625_111354_per_class_metrics.csv")

    thresholds = (
        thresholds[thresholds["model_split"] == "val_cv"]["threshold"]
        .reset_index()
        .drop(columns="index")
    )
    thr = thresholds.to_numpy().ravel()
    print(f"Thresholds (ndarray) shape: {thr.shape}")

    probs = {}

    for label, model in models.items():
        probs[label] = model.predict_proba(X_scaled)[:, 1]  # positive class prob


    probs_df = pd.DataFrame(probs)
    print(probs_df.head())
    probs_array = probs_df.to_numpy()
    print(f"Probs (ndarray) shape: {probs_array.shape}")

    preds = (probs_array >= thr).astype(int)
    preds = pd.DataFrame(preds, columns=probs_df.columns, index=probs_df.index)
    print(f"Preds (DataFrame) shape: {preds.shape}")

    label_lists = preds.apply(
        lambda row: list(row[row == 1].index),
        axis=1
    )

    val_scores = score(
        y,
        probs_array,
        class_names=mlb.classes_,
        thresholds=thr,
    )
    all_metrics = {
        "val": val_scores
    }

    export_metrics_csvs(all_metrics, "./validation/models/main")
    plot_binary_confusion_heatmap(
        y, probs_df.to_numpy(),
        mlb.classes_,
        thr,
        save_path="./validation/models/main/confusion_heatmap.png"
        )

