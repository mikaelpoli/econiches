"""
Stratified train/test (or test/val) split for MAG datasets.

Splits are performed at the metagenome level (i.e. all MAGs from the same
metagenome land in the same partition) while keeping each class's fraction in
the held-out set as close as possible to `--pct`.

Usage
-----
# Create train / test split using 'environment' labels, 30 % test:
    python split_metagenomes.py

# Create test / val split on a previously saved test set, 20 % val:
    python split_metagenomes.py --val --pct 0.2

# Use macro-environment labels instead:
    python split_metagenomes.py --environment macro_environment
"""

import argparse
import random
from pathlib import Path

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import econiches.utils as utils

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = Path("/home/simple/Users/mikael/econiches")
PLOTS_ROOT     = BASE_DIR / "validation" / "plots" / "split"
FILTERED_FULL  = BASE_DIR / "validation" / "data" / "filtered" / "full"
FILTERED_SPLIT = BASE_DIR / "validation" / "data" / "filtered" / "split"


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def split_metagenomes(
    df: pd.DataFrame,
    *,
    class_col: str,
    metagenome_id_col: str,
    test_fraction: float = 0.2,
    n_restarts: int = 20,
    plateau_tol: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[float]]:
    """
    Assign each row a 'split' label ('train' or 'test') by greedily shuffling
    metagenomes across restarts and picking the assignment whose per-class
    test fractions are closest to `test_fraction`.

    All MAGs that belong to the same metagenome are always kept together,
    preventing data leakage between splits.

    The optimisation minimises a weighted squared error between the actual and
    target per-class counts in the held-out set, where rare classes receive
    higher weight so they are not drowned out by dominant ones.

    Parameters
    ----------
    df : pd.DataFrame
        Row-per-MAG table that must contain at least `metagenome_id_col` and
        `class_col` columns.
    class_col : str
        Column name holding the class label (e.g. 'environment').
    metagenome_id_col : str
        Column name holding the metagenome identifier. All MAGs sharing the
        same identifier are kept in the same split partition.
    test_fraction : float, optional
        Target fraction of each class to place in the held-out set.
        Default is 0.2.
    n_restarts : int, optional
        Maximum number of random shuffles to try. A good rule of thumb is to
        set this to the number of unique metagenomes. Default is 20.
    plateau_tol : int, optional
        Stop early if the best cost has not improved for this many consecutive
        restarts. Default is 10.
    seed : int, optional
        Base random seed. Restart i uses seed + i, so results are fully
        reproducible. Default is 0.

    Returns
    -------
    df : pd.DataFrame
        Copy of the input with an added 'split' column containing 'train' or
        'test' for every row.
    cost_history : list[float]
        Best (lowest) cost recorded each time a new optimum was found, one
        entry per improvement. Useful for plotting convergence.
    """
    print(f"Splitting: test_fraction={test_fraction}, restarts={n_restarts}, seed={seed}")

    # Build a (metagenomes × classes) count matrix
    pivot       = df.groupby([metagenome_id_col, class_col]).size().unstack(fill_value=0)
    metagenomes = pivot.index.tolist()
    counts      = pivot.values                                  # shape: (M, C)

    total       = counts.sum(axis=0)                            # total per class
    target      = np.ceil(total * test_fraction).astype(int)    # desired test count per class

    # Weighted squared error — rare classes matter as much as common ones
    weights = 1.0 / (total / len(df) + 1e-6)
    cost    = lambda vec: np.sum(((vec - target) * weights) ** 2)

    best_split: set = set()
    best_cost = float("inf")
    cost_history: list[float] = []
    plateau = 0

    with tqdm(total=n_restarts, desc="Optimising split", unit="restart") as pbar:
        for restart in range(n_restarts):
            random.seed(seed + restart)
            order = list(range(len(metagenomes)))
            random.shuffle(order)

            current = np.zeros_like(target)
            test_set: set = set()
            for i in order:
                candidate = current + counts[i]
                if cost(candidate) < cost(current):
                    test_set.add(metagenomes[i])
                    current = candidate

            c = cost(current)
            if c < best_cost:
                best_cost, best_split = c, test_set
                cost_history.append(best_cost)
                plateau = 0
            elif c - best_cost < 1e-3:
                plateau += 1
                if plateau >= plateau_tol:
                    pbar.set_postfix_str(f"early stop — no improvement for {plateau_tol} restarts")
                    pbar.update(n_restarts - restart) # jump bar to 100 %
                    break

            pbar.set_postfix(best_cost=f"{best_cost:.4f}", plateau=plateau)
            pbar.update()

    df = df.copy()
    df["split"] = df[metagenome_id_col].map(lambda mg: "test" if mg in best_split else "train")
    return df, cost_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(
    df: pd.DataFrame,
    *,
    class_col: str,
    split_name: str, # label used for the held-out partition, e.g. "test" or "val"
    test_fraction: float,
    save_path: Path,
) -> pd.DataFrame:
    """
    Print a per-class summary table and save a bar chart showing how close
    each class's held-out fraction is to the target.

    Each bar represents one class, coloured by class size (Blues colormap).
    A horizontal line marks the target fraction. Ratio values are annotated
    above each bar. The figure is saved at 300 dpi.

    Parameters
    ----------
    df : pd.DataFrame
        Row-per-MAG table with at least a `class_col` column and a 'split'
        column (as produced by `split_metagenomes`).
    class_col : str
        Column name holding the class label (e.g. 'environment').
    split_name : str
        Value in the 'split' column that identifies the held-out partition,
        e.g. 'test' or 'val'.
    test_fraction : float
        Target held-out fraction; used to compute per-class target counts and
        to draw the reference line on the chart.
    save_path : Path
        File path where the figure is saved (PNG recommended).

    Returns
    -------
    summary : pd.DataFrame
        Per-class DataFrame with columns:
        - 'total'               - total sample count
        - '\\<split_name>'       - samples in the held-out partition
        - 'target_\\<split_name>'- target count (ceil of total x test_fraction)
        - '\\<split_name>_ratio' - actual held-out fraction
        - 'target_ratio'        - target fraction (target count / total)
        - 'diff_from_target'    - actual ratio minus target ratio
    """
    total  = df.groupby(class_col).size()
    held   = df[df["split"] == split_name].groupby(class_col).size()
    target = np.ceil(total * test_fraction).astype(int)
    ratios = held / total

    summary = pd.DataFrame({
        "total":           total,
        split_name:        held,
        f"target_{split_name}": target,
        f"{split_name}_ratio":  ratios,
        "target_ratio":    target / total,
    })
    summary["diff_from_target"] = summary[f"{split_name}_ratio"] - summary["target_ratio"]

    print("\n=== Split Quality Summary ===")
    print(summary.sort_values("diff_from_target").round(3))
    n_held = held.sum()
    n_total = total.sum()
    print(f"\nTrain: {n_total - n_held} | {split_name.title()}: {n_held} | "
          f"Overall ratio: {n_held / n_total:.3f} | "
          f"Median {ratios.median():.3f} | Mean: {ratios.mean():.3f} | Std: {ratios.std():.3f}")

    # Bar chart coloured by class size
    ratios_sorted = ratios.sort_values()
    sizes  = summary["total"].loc[ratios_sorted.index].values
    norm   = mpl.colors.Normalize(vmin=sizes.min(), vmax=sizes.max())
    cmap   = plt.cm.Blues

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(range(len(ratios_sorted)), ratios_sorted.values,
                  color=cmap(norm(sizes)), edgecolor="black")
    ax.axhline(test_fraction, color="black", linewidth=1.5)
    ax.set_xticks(range(len(ratios_sorted)))
    ax.set_xticklabels(ratios_sorted.index, rotation=45, ha="right")
    ax.set_ylabel(f"{split_name.title()} Ratio per Class")
    ax.set_title(f"Split Quality — {split_name.title()} (target {test_fraction:.0%})")
    ax.set_ylim(0, 1)
    for bar, r in zip(bars, ratios_sorted.values):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                f"{r:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax).set_label("Class size (total samples)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    parser.add_argument(
        "-e", "--environment",
        default="environment",
        choices=["environment", "macro_environment"],
        help="Column to use as the class label (default: 'environment')"
        )
    parser.add_argument(
        "-p", "--pct",
        type=float,
        default=0.2,
        help="Fraction of data to hold out (default: 0.2)"
        )
    parser.add_argument(
        "-v", "--val",
        action="store_true",
        help="Split the existing test set into test/val instead of train/test"
        )
    args = parser.parse_args()

    ENV        = args.environment
    PCT        = args.pct
    IS_VAL     = args.val
    SPLIT_NAME = "val" if IS_VAL else "test"

    # Resolve input paths
    if IS_VAL:
        features_file = FILTERED_SPLIT / ENV / "train_test" / "test" / "X_test.parquet"
        labels_file   = FILTERED_SPLIT / ENV / "train_test" / "test" / "y_test.csv"
        split_a, split_b = "test", "val"
    else:
        features_file = FILTERED_FULL / "features.parquet"
        labels_file   = FILTERED_FULL / "labels.csv"
        split_a, split_b = "train", "test"

    plots_dir = PLOTS_ROOT / ENV
    plots_dir.mkdir(parents=True, exist_ok=True)
    for d in (FILTERED_FULL, FILTERED_SPLIT / ENV):
        d.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    features = pd.read_parquet(features_file).set_index("mag_id").sort_index()
    labels   = (
        pd.read_csv(labels_file, usecols=["mag_id", "metagenome_id", ENV])
        .sort_values("mag_id")
        .apply(lambda c: c.astype("category") if c.dtype == "object" else c)
    )
    labels = labels[labels["mag_id"].isin(features.index)]

    missing = features.index.difference(labels["mag_id"])
    print(f"Loaded {features_file}\n"
          f"Shape: {features.shape} | Unique MAGs: {features.index.nunique()} | "
          f"Missing {ENV} labels: {len(missing)}")

    # --- Split ---
    n_metagenomes = labels["metagenome_id"].nunique()
    split_df, cost_history = split_metagenomes(
        labels,
        class_col=ENV,
        metagenome_id_col="metagenome_id",
        test_fraction=PCT,
        n_restarts=n_metagenomes,
        plateau_tol=20,
        seed=0,
    )

    # Remap split labels for the val stage (train→test, test→val)
    if IS_VAL:
        split_df["split"] = split_df["split"].map({"train": "test", "test": "val"})

    # --- Sanity checks ---
    mags_a = set(split_df[split_df["split"] == split_a]["mag_id"])
    mags_b = set(split_df[split_df["split"] == split_b]["mag_id"])
    assert mags_a.isdisjoint(mags_b), f"Data leak: MAGs appear in both {split_a} and {split_b}!"
    print(f"{split_a.title()} MAGs: {len(mags_a)} | {split_b.title()} MAGs: {len(mags_b)}")

    # Optimization convergence plot
    plt.plot(cost_history, marker="o")
    plt.xlabel("Restart"); plt.ylabel("Cost"); plt.title("Split Optimization Cost")
    plt.savefig(plots_dir / f"{SPLIT_NAME}_split_cost.png"); plt.close()

    # Quality evaluation plot
    evaluate_split(split_df, class_col=ENV, split_name=split_b, test_fraction=PCT,
                   save_path=plots_dir / f"{SPLIT_NAME}_split_quality.png")

    # --- Export ---
    split_map  = split_df.set_index("mag_id")[["split"]].loc[lambda d: ~d.index.duplicated()]
    X_features = features.merge(split_map, left_index=True, right_index=True).reset_index(names="mag_id")
    X_labels   = labels.set_index("mag_id").join(split_map).reset_index(names="mag_id")

    stage_dir  = FILTERED_SPLIT / ENV / ("test_val" if IS_VAL else "train_test")

    for name in (split_a, split_b):
        out_dir = stage_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        feat_out  = out_dir / f"X_{name}.parquet"
        label_out = out_dir / f"y_{name}.csv"

        (X_features[X_features["split"] == name]
            .drop(columns="split")
            .to_parquet(feat_out, engine="pyarrow", compression="snappy", index=False))

        (X_labels[X_labels["split"] == name]
            .drop(columns="split")
            .to_csv(label_out, index=False))

        print(f"Saved {name}: {feat_out}, {label_out}")