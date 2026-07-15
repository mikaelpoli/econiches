# ==========================================================
# Imports:
#     * Labels: 'mag_labels.csv'
#     * Parsed eggNOG-mapper v2 MAG annotations (format: parquet),
#       containing the per-MAG raw counts of all terms in:
#       - EC
#       - COG_category
#       - KEGG_ko
#       - CAZy
#       - PFAM
# ==========================================================
# - Produces:
#     * EDA for labels and annotations DataFrames
# ==========================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from pathlib import Path
import seaborn as sns
import econiches.utils as utils


def _format_variable_label(
    variable: str,
    capital_prefixes: set[str] = {"CAZy", "COG", "EC", "KEGG", "PFAM"},
) -> str:
    """Convert a variable name like 'CAZy_family' to a readable label."""

    parts = variable.split("_")
    prefix, *rest = parts

    base = prefix if prefix in capital_prefixes else prefix.title()
    suffix = " ".join(rest).title()

    return f"{base} {suffix}".strip()


def plot_bioindex(
        df: pd.DataFrame,
        x: str,
        y: str,
        savefilename: Path | None = None,
        save: bool = True,
        show: bool = False
):
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = df.groupby(x)[y].apply(list)

    ax.boxplot(groups.values)
    ax.set_xticklabels(groups.index, rotation=45, ha="right")

    xlab = _format_variable_label(x)
    ylab = _format_variable_label(y)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"{ylab} by {xlab}")

    plt.tight_layout()

    if save and savefilename:
        plt.savefig(savefilename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    plt.close(fig)


def plot_distribution_univariate(
    counts: pd.Series,
    variable: str,
    percent: bool = False,
    min_mag: int = 0,
    savefilename: Path | None = None,
    save: bool = True,
    show: bool = False,
):
    """Plot a univariate distribution from a counts series."""
    counts_plot = counts.copy()

    if percent:
        total = counts_plot.sum()
        counts_plot = counts_plot / total
        min_mag = min_mag / total if min_mag else 0

    lab = _format_variable_label(variable)

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.barplot(
        x=counts_plot.values,
        y=counts_plot.index,
        ax=ax
    )

    ax.xaxis.tick_top()
    ax.set_ylabel(None)

    title = (
        f"Relative Distribution of {lab}"
        if percent else f"Distribution of {lab}"
    )
    fig.suptitle(title, fontsize=14)

    if percent:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))

    if min_mag:
        ax.axvline(x=min_mag, color="black")

    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()

    if save and savefilename:
        fig.savefig(savefilename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)


def plot_heatmap(
        df: pd.DataFrame,
        var1: str,
        var2: str,
        by_row: bool = True,
        cmap="Blues",
        savefilename: Path | None = None,
        save: bool = True,
        show: bool = False
):
    table = (
        df
        .groupby([var2, var1])
        .size()
        .unstack(fill_value=0)
    )
    if by_row:
        table_norm = table.div(table.sum(axis=1), axis=0)
        # xlab = _format_variable_label(var1)
        # ylab = _format_variable_label(var2)
    else:
        table_norm = table.div(table.sum(axis=0), axis=1)
        # xlab = _format_variable_label(var2)
        # ylab = _format_variable_label(var1)

    plt.figure(figsize=(12,10))
    sns.heatmap(table_norm, cmap=cmap)

    xlab = _format_variable_label(var1)
    ylab = _format_variable_label(var2)

    plt.title(f"Relative Abundance of {ylab} Across {xlab}")
    plt.xlabel(f"{xlab}")
    plt.ylabel(f"{ylab}")
    plt.tight_layout()

    if save and savefilename:
        plt.savefig(savefilename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()


def plot_barplot_counts(
        df: pd.DataFrame,
        var1: str,
        var2: str,
        relative: bool = True,
        stacked: bool = True,
        rot: int = 0,
        savefilename: Path | None = None,
        save: bool = True,
        show: bool = False,
        title: str | None = None
):
    counts = (
        df
        .groupby([var2, var1])
        .size()
        .unstack(fill_value=0)
    )

    if relative:
        counts = counts.div(counts.sum(axis=1), axis=0)
    
    counts.plot.bar(stacked=stacked, figsize=(10,6))

    dep = _format_variable_label(var1)
    indep = _format_variable_label(var2)
    ylab = "Relative Freq" if relative else "Counts"

    plt.xlabel(indep)
    plt.ylabel(ylab)

    title = title if title is not None else f"{dep} Distribution by {indep}"

    plt.title(title)
    plt.xticks(rotation=rot, ha="right")

    plt.tight_layout()

    if save and savefilename:
        plt.savefig(savefilename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()


def plot_barplot_counts_stacked_and_grouped(df, savefilename=None):
    env_order = df["environment"].unique()
    part_order = ["Train", "Test"]

    df["environment"] = pd.Categorical(df["environment"], categories=env_order, ordered=True)
    df["Partition"] = pd.Categorical(df["Partition"], categories=part_order, ordered=True)

    df = df.sort_values(["environment", "Partition"])

    arch = df.pivot(index="environment", columns="Partition", values="Archaea")
    bact = df.pivot(index="environment", columns="Partition", values="Bacteria")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(arch.index))
    width = 0.30

    # Train bars (left)
    ax.bar([i - width/2 for i in x],
        arch["Train"],
        width,
        label="Archaea (Train)",
        color="tab:blue")

    ax.bar([i - width/2 for i in x],
        bact["Train"],
        width,
        bottom=arch["Train"],
        label="Bacteria (Train)",
        color="tab:orange")

    # Test bars (right)
    ax.bar([i + width/2 for i in x],
        arch["Test"],
        width,
        label="Archaea (Test)",
        color="tab:blue",
        alpha=0.5)

    ax.bar([i + width/2 for i in x],
        bact["Test"],
        width,
        bottom=arch["Test"],
        label="Bacteria (Test)",
        color="tab:orange",
        alpha=0.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(arch.index, rotation=90)
    ax.set_ylabel("Proportion")
    ax.set_title("Archaea vs Bacteria by Environment (Train vs Test)")
    
    plt.xticks(rotation=45, ha="right")
    max_prop = df["Archaea"].to_numpy().max()

    ax.legend(ncol=2, loc="upper center", framealpha=0.90)

    ax.axhline(
        y=max_prop,
        linestyle="--",
        color="black",
        linewidth=1,
    )

    plt.tight_layout()

    if savefilename:
        plt.savefig(savefilename, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()




def plot_feature_var_vs_prev(
        var: pd.Series,
        prev: pd.Series,
        log_var: bool = False,
        var_threshold: float = None,
        prev_threshold: float = None,
        label_outliers: bool = False,
        iqr_multiplier: float = 1.5,
        prefixes: dict = None,
        log_lab: bool = True,
        ylab: str = None,
        title: str = None,
        save: bool = True,
        savefilename: Path | None = None,
        show: bool = False
):
    plot_df = pd.DataFrame({
        "v": var,
        "p": prev
    })

    # Detect outliers
    if label_outliers:
        Q1 = var.quantile(0.25)
        Q3 = var.quantile(0.75)
        IQR = Q3 - Q1
        lb = Q1 - iqr_multiplier * IQR
        ub = Q3 + iqr_multiplier * IQR
        outlier_df = plot_df[(plot_df["v"] < lb) | (plot_df["v"] > ub)]

    plt.figure(figsize=(8,6))

    if prefixes:
        plot_df["type"] = [utils._assign_feature_family(col, prefixes) for col in plot_df.index]
        for t in plot_df["type"].unique():
            subset = plot_df[plot_df["type"] == t]
            plt.scatter(subset["p"],
                        subset["v"],
                        alpha=0.2,
                        label=t)
        plt.legend()
    else:
        plt.scatter(prev, var, alpha=0.5)

    # Label outliers
    if label_outliers:
        for feature, row in outlier_df.iterrows():
            plt.text(row["p"], row["v"], feature, fontsize=8, alpha=0.7)

    prev_med = prev.median()
    var_med = var.median()

    plt.axvline(prev_med, linestyle='--')
    plt.axhline(var_med, linestyle='--')

    plt.text(prev_med, plt.ylim()[1], f"median ({prev_med:.2f})",
            rotation=90, va='top', ha='right')

    plt.text(plt.xlim()[1], var_med, f"median ({var_med:.2f})",
            va='bottom', ha='right')
    
    if log_var:
        plt.yscale('log')

    plt.xlabel("Prevalence")
    if ylab:
        plt.ylabel(ylab)
    else:
        plt.ylabel(f"Variance {'(log)' if log_lab else ''}")
    if title:
        plt.title(title)
    else:
        plt.title(f"Variance {'(log)' if log_lab else ''} vs Prevalence {'by Feature Type' if prefixes else ''}")
    plt.tight_layout()

    if save and savefilename:
        plt.savefig(savefilename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    plt.close()