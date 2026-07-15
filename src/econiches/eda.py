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

from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import econiches.utils as utils
import econiches.plot_fn as pf


def shannon_index(
        df: pd.DataFrame,
        family_cols: dict
) -> float:
    """
    Quantifies the uncertainty in predicting the species/term identity 
    of an observation that is taken at random from the dataset.
    """
    res = pd.DataFrame(index=df.index)

    for family, cols in family_cols.items():
        X = df[cols].to_numpy(dtype=np.float32)
        X = np.log1p(X) # Reflects variety rather than volume (i.e. log()); prevents /0

        row_sums = X.sum(axis=1, keepdims=True)

        # Calculate probabilities
        P = np.zeros_like(X)
        mask = row_sums.squeeze() > 0
        P[mask] = X[mask] / row_sums[mask]

        # Shannon entropy
        H = -np.sum(P * np.log(P + 1e-12), axis=1)

        res[f"{family}_shannon"] = H

    return res


def annotation_richness(
        df: pd.DataFrame, 
        family_cols: dict
) -> float:
    res = pd.DataFrame(index=df.index)
    for family, cols in family_cols.items():
        mask = (df[cols] != 0).sum(axis=1)
        res[f"{family}_richness"] = mask
    return res


def annotation_evenness(
        shannon_df: pd.DataFrame,
        family_cols: dict
) -> float:
    res = pd.DataFrame(index=shannon_df.index)

    for family, cols in family_cols.items():
        N = len(cols)
        res[f"{family}_evenness"] = shannon_df[f"{family}_shannon"] / np.log(N)

    return res


def compute_feature_prevalence(df: pd.DataFrame):
    select_df = df.select_dtypes(include="number")
    prevalence = (select_df > 0).mean(axis=0)
    return prevalence


def compute_cv(df: pd.DataFrame, log_transform: bool = True, epsilon: float = 1e-8):
    select_df = df.select_dtypes(include="number")
    if log_transform:
        select_df = np.log1p(select_df)
    mean = select_df.mean(axis=0)
    std = select_df.std(axis=0)
    return std / (mean + epsilon)


def filter_groupwise(df: pd.DataFrame, group_col:str = "environment", protected_cols:list = None):
    if protected_cols is None:
        protected_cols = []
    protected_cols = set(protected_cols + [group_col])

    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols = [col for col in numeric_cols if col not in protected_cols]

    nonzero_mask = df[numeric_cols] != 0 # True if value > 0
    common_mask = nonzero_mask.groupby(df[group_col]).all() # True if column is non-zero in all MAGs of the group
    common_cols = common_mask.all() # True if column is non-zero in all groups
    keep_cols = protected_cols | set(common_cols[~common_cols].index)

    return df[list(keep_cols)]


# =============================
# MAIN PROGRAM
# =============================
if __name__ == "__main__":
    # -----------------------------
    # CONFIGURATION
    # -----------------------------
    from pathlib import Path

    BASE_DIR        = Path("./")
    SRC_DIR         = BASE_DIR / 'src'
    utils.append_wd(SRC_DIR)
    PLOTS_DIR       = BASE_DIR / 'plots' / 'full'
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR      = BASE_DIR / 'parsed'
    COVERAGE_DIR    = BASE_DIR / 'coverage'
    STATS_DIR       = BASE_DIR / 'data' / 'stats'
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FILTERED_DIR     = BASE_DIR / 'filtered'
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    LABELS_FILE     = PARSED_DIR / 'mag_labels.csv'
    # COVERAGE_FILE   = COVERAGE_DIR / 'coverage_all_mags.csv'
    PARSED_FILE     = PARSED_DIR / "mag_features.parquet"
    FAMILY_PREFIXES = {
        "EC": ["EC_"],
        "COG": ["COG_"],
        "KEGG": ["KO_"],
        "PFAM": ["PFAM_"],
        "CAZy": ["CAZy_"]
        }
    ENV_TYPES      = ["macro_environment", "environment", "sub_environment"]
    TAX_COLS       = [
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species"
    ]
    META_INFO      = ["genome_size", "metagenome_id", 
                      "sequential_id", "file_extension", 
                      "GA", "OTU"]
    
    MIN_MAG         = 20

    # -----------------------------
    # IMPORT DATA
    # -----------------------------
    annotations    = pd.read_parquet(PARSED_FILE).sort_index()

    labels         = pd.read_csv(LABELS_FILE, sep=',')
    labels         = labels.apply(
        lambda col: col.astype("category") if col.dtype == "object" else col
        )
    
    labels         = labels[labels["mags"].isin(annotations.index)]

    annotations    = annotations.join(labels.set_index("mags"), how="left")

    FAMILY_COLS    = {
        family: [c for c in annotations.columns if any(c.startswith(p) for p in prefixes)]
        for family, prefixes in FAMILY_PREFIXES.items()
        }
    
    # coverage       = pd.read_csv(COVERAGE_FILE, sep=",")

    dfs = defaultdict()

    # -----------------------------
    # CHECK DATA CORRECTNESS
    # -----------------------------
    print(f"Loaded dataset: {PARSED_FILE}")
    dfs["og"] = copy.deepcopy(annotations)

    utils.print_data_check_header()

    CHECK_LABS = dfs["og"].index.difference(labels["mags"])
    CHECK_DUP = dfs["og"].duplicated()

    print(f"{'All MAGs labeled' if CHECK_LABS.empty else 'Missing MAG labels'}")
    print(f"Found {CHECK_DUP.sum()} duplicates")

    print(f"Shape: {dfs["og"].shape}")
    print(f"Unique MAGs: {dfs["og"].index.nunique()}\n")
    dfs["og"].info()

    utils.print_section_separator()

    # -----------------------------
    # EXPLORATORY DATA ANALYSIS
    # -----------------------------
    # Coverage
    # print("Annotation coverage (%):")
    # print("\n")
    # print(f"{coverage.describe()}")

    # utils.print_section_separator()

    print(f"Dropping environments with less than {MIN_MAG} MAGs...")
    MIN_MAG_KEY = f"min_mags_{MIN_MAG}"

    dfs[MIN_MAG_KEY] = dfs["og"].groupby("environment").filter(lambda x: len(x) >= MIN_MAG)
    _env = set(dfs["og"]["environment"].unique())
    _env_keep = set(dfs[MIN_MAG_KEY]["environment"])
    _env_drop = list(_env.difference(_env_keep))

    print(f"Dropped {len(_env_drop)} environments:")
    n = 5
    print(*[" | ".join(map(str, _env_drop[i:i+n])) for i in range(0, len(_env_drop), n)], sep="\n")

    utils.print_data_check_header()

    print(f"Shape: {dfs[MIN_MAG_KEY].shape}")
    print(f"Unique MAGs: {dfs[MIN_MAG_KEY].index.nunique()}\n")
    dfs[MIN_MAG_KEY].info()

    utils.print_section_separator()

    # Drop features with 0% and 100% prevalence
    print("Dropping features with 0% prevalence...")
    zero_prev = (dfs[MIN_MAG_KEY].sum(axis=0) == 0)
    dfs[MIN_MAG_KEY] = dfs[MIN_MAG_KEY].loc[:, ~zero_prev]
    print(f"Dropped {zero_prev.sum()} features")

    utils.print_data_check_header()

    print(f"Shape: {dfs[MIN_MAG_KEY].shape}")
    print(f"Unique MAGs: {dfs[MIN_MAG_KEY].index.nunique()}\n")
    dfs[MIN_MAG_KEY].info()

    utils.print_section_separator()

    print(f"Dropping features with 100% prevalence...")
    _cols_protect = META_INFO + ENV_TYPES + TAX_COLS
    dfs["filtered"] = filter_groupwise(dfs[MIN_MAG_KEY], protected_cols=_cols_protect)
    _dropped_cols = set(dfs[MIN_MAG_KEY].columns).difference(set(dfs["filtered"].columns))
    print(f"Dropped {len(_dropped_cols)} features")

    PRINT_COMMON_COLS = True
    if PRINT_COMMON_COLS:
        n = 5
        _dropped_cols_list = list(_dropped_cols)
        print(*[" | ".join(map(str, _dropped_cols_list[i:i+n])) for i in range(0, len(_dropped_cols_list), n)], sep="\n")

    utils.print_data_check_header()

    print(f"Shape: {dfs["filtered"].shape}")
    print(f"Unique MAGs: {dfs["filtered"].index.nunique()}")
    print(f"Unique metagenomes: {dfs["filtered"]["metagenome_id"].nunique()}")
    print(f"Unique environments: {dfs["filtered"]["environment"].nunique()}")
    dfs["filtered"].info()

    utils.print_section_separator()

    # Check missing taxonomic information
    print("Missing taxonomic information:")
    _tax_nan_counts = {f"{col.title()} NaN": int(dfs["filtered"][col].isna().sum()) for col in TAX_COLS}
    for k, v in _tax_nan_counts.items():
        print(f"{k}: {v}")
    
    utils.print_section_separator()

    # Check counts per macro environment and environment
    macro_env_vc = dfs["filtered"]["macro_environment"].value_counts()
    env_vc = dfs["filtered"]["environment"].value_counts()

    print("Macro environment counts:")
    print(f"{macro_env_vc}")
    print("\n")

    print("Environment counts:")
    print(f"{env_vc}")

    utils.print_section_separator()

    # Check multilabel MAGs counts
    multilabel_counts = dfs["filtered"].index.value_counts()
    multilabel_counts.sort_values(ascending=False, inplace=True)

    print(f"Multilabel MAGs: {(multilabel_counts > 1).sum()}")
    print(f"{multilabel_counts.describe()}")

    # OPTIONAL: Filter out multilabel MAGs
    REMOVE_MULTILABEL = False
    if REMOVE_MULTILABEL:
        dfs["single_label"] = dfs["filtered"][~dfs["filtered"].index.isin(multilabel_counts[multilabel_counts > 1].index)]

        utils.print_data_check_header()

        print(f"Shape: {dfs["single_label"].shape}")
        print(f"Unique MAGs: {dfs["single_label"].index.nunique()}\n")
        print(f"Unique metagenomes: {dfs["single_label"]["metagenome_id"].nunique()}")
        print(f"Unique environments: {dfs["single_label"]["environment"].nunique()}")
        dfs["filtered"].info()

        utils.print_section_separator()

        # Check counts per macro environment and environment
        macro_env_vc_sl = dfs["single_label"]["macro_environment"].value_counts()
        env_vc_sl = dfs["single_label"]["environment"].value_counts()

        print("Macro environment counts:")
        print(f"{macro_env_vc_sl}")
        print("\n")

        print("Environment counts:")
        print(f"{env_vc_sl}")

    utils.print_section_separator()

    # MAG distribution within metagenomes by environment
    metagenome_counts = dfs["filtered"].groupby("environment")["metagenome_id"].agg(
        Metagenomes="nunique",
        MAGs="count"
    )
    metagenome_counts.sort_values(by=["Metagenomes", "MAGs"], inplace=True)

    PLOT = False
    if PLOT:
        df = dfs["filtered"]
        environments = list(df["environment"].unique())
        mag_counts = {}
        for env in environments:
            mag_counts[env] = df[df["environment"] == env]["metagenome_id"].value_counts()

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x=mag_counts[env].index, y=mag_counts[env].values, ax=ax)
            ax.set_title(f"Metagenome ID counts - {env}")
            ax.set_xlabel("Metagenome ID")
            ax.set_ylabel("Count")
            ax.set_xticks([])

            plt.tight_layout()
            DIR = PLOTS_DIR / "split_diagnostics"
            DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(DIR / f"{env}_barplot.png", dpi=150)
            plt.close()

    # Compute 20% test set size
    metagenome_counts["n_test_20pct"] = np.ceil(metagenome_counts["MAGs"] / 5).astype(int)

    print("MAG distribution within metagenomes by environment:")
    print(f"{metagenome_counts}")

    SAVE = False
    if SAVE:
        metagenome_counts.to_csv(STATS_DIR / "metagenome_counts.csv")

    utils.print_section_separator()

    # Check number of features per annotation family
    print("Number of features per annotation family:")
    FAMILY_COLS = {
        family: [c for c in dfs["filtered"].columns if any(c.startswith(p) for p in prefixes)]
        for family, prefixes in FAMILY_PREFIXES.items()
    }

    _, ncol = dfs["filtered"].shape

    for family, cols in FAMILY_COLS.items():
        print(f"{family} = {len(cols)} ({(len(cols)/ncol)*100:.6f}%)")

    utils.print_section_separator()

    # Compute feature prevalence and coefficient of variation
    _cols_exclude = set(META_INFO + ENV_TYPES + TAX_COLS)
    prev = compute_feature_prevalence(dfs["filtered"].drop(columns=_cols_exclude, inplace=False))
    cv = compute_cv(dfs["filtered"].drop(columns=_cols_exclude, inplace=False), log_transform=False)

    print("Feature prevalence statistics:")
    print(prev.describe())
    print("\n")

    print("Feature CV statistics:")
    print(cv.describe())

    PLOT = True
    if PLOT:
        VAR_PLOTS_DIR = PLOTS_DIR / "var"
        VAR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        pf.plot_feature_var_vs_prev(
            cv,
            prev,
            var_threshold=cv.median(),
            prev_threshold=0.5,
            savefilename=VAR_PLOTS_DIR / "cv_vs_prevalence.png",
            ylab="CV",
            title="CV vs Prevalence"
            )
        
    utils.print_section_separator()

    ECO_METRICS = False
    BY_DOMAIN = False
    if ECO_METRICS:
        print("Computing ecological metrics...")
        COG_TERMS = ["COG_S", "COG_Q"]

        for name, df in dfs.items():
            family_cols = defaultdict(list)
            for family, prefix in FAMILY_PREFIXES.items():
                family_cols[family] = utils._get_family_cols(df, prefix[0])

            shannon = shannon_index(df, family_cols)
            richness = annotation_richness(df, family_cols)

            df = pd.concat([df, shannon, richness], axis=1)

            evenness = annotation_evenness(df, family_cols)

            dfs[name] = pd.concat([df, evenness], axis=1)

            for term in COG_TERMS:
                cog_cols = [c for c in df.columns if "COG_" in c]
                if term in list(df.columns) and cog_cols:
                    dfs[name][f"{term}_abundance"] = dfs[name][term] / dfs[name][cog_cols].sum(axis=1)
        
        print("Ecological metrics computed")
        
        utils.print_section_separator()

        # Domain-level analysis
        if BY_DOMAIN:
            print("Plotting domain-level information...")

            DOMAIN_PLOTS_DIR = PLOTS_DIR / "domain"
            DOMAIN_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            df = dfs["filtered"]

            for family, prefix in FAMILY_PREFIXES.items():
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    data=df,
                    x="genome_size",
                    y=f"{str(family)}_richness",
                    hue="domain",
                    ax=ax)
                
                plt.savefig(DOMAIN_PLOTS_DIR / f"{str(family)}_richness_by_genome_size.png", format="png")
            
            print("Finished plotting")

            utils.print_section_separator()

    # -----------------------------
    # EXPORT DATA
    # -----------------------------
    # Save labels
    SAVE = False
    if SAVE:
        df = dfs["filtered"]
        _save_cols = ["metagenome_id", "macro_environment", "environment"]
        labels = df[_save_cols]

        print(f"Saving labels...")
        save_dir =  FILTERED_DIR / "full"
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir/ "labels.csv"
        labels.to_csv(filename, index=True, index_label="mag_id")
        print(f"Labels saved to {filename}")

        utils.print_section_separator()

    # Save features
    SAVE = False
    if SAVE:
        meta_info = [c for c in META_INFO if c != "metagenome_id"]
        base_cols = meta_info + ENV_TYPES + TAX_COLS

        if ECO_METRICS:
            suffixes = [
                "_shannon",
                "_richness",
                "_evenness",
                "_abundance",
            ]

            eco_metrics = [
                c for c in dfs["filtered"].columns
                if any(c.endswith(suffix) for suffix in suffixes)
            ]
            base_cols = base_cols + eco_metrics

        _drop_cols = set(base_cols)
        df = dfs["filtered"].drop(columns=_drop_cols, inplace=False)

        print(f"Saving feature dataset...")
        save_dir =  FILTERED_DIR / "full"
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / "features.parquet"
        df.index.name = "mag_id"
        df.reset_index(inplace=True)
        df.to_parquet(
            filename,
            engine="pyarrow",
            compression="snappy",
            index=False
        )
        print(f"Saved features to {filename}")

        utils.print_section_separator()

    
    # -----------------------------
    # PLOTS
    # -----------------------------
    PLOT = False
    if PLOT:
        print("Plotting ecological metrics...")

        ENV_TYPES.remove("sub_environment")
        df = dfs["filtered"]
        SAVE_DIR = PLOTS_DIR / "filtered"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        # Shannon, richness, evenness by environment type
        for family in FAMILY_PREFIXES:
            for env in ENV_TYPES:
                for metric in ["shannon", "richness", "evenness"]:
                    pf.plot_bioindex(
                        df, env, f"{family}_{metric}",
                        savefilename=SAVE_DIR / f"{family}_{metric}_by_{env}.png"
                    )

        # Environment and macro environment distributions
        for env, vc_clean in zip(ENV_TYPES, [env_vc, macro_env_vc]):
            pf.plot_distribution_univariate(
                vc_clean, env,
                savefilename=SAVE_DIR / f"{env}_count_distribution.png"
            )
            pf.plot_distribution_univariate(
                vc_clean, env, percent=True,
                savefilename=SAVE_DIR / f"{env}_percent_distribution.png"
            )

        # Phylum abundance heatmaps
        for env in ENV_TYPES:
            pf.plot_heatmap(
                df, "phylum", env,
                savefilename=SAVE_DIR / f"phyla_by_{env}.png"
            )

        # Genome size bioindex
        for env in ENV_TYPES:
            pf.plot_bioindex(
                df, env, "genome_size",
                savefilename=SAVE_DIR / f"genome_size_by_{env}.png"
            )

        # Domain counts
        for env in ENV_TYPES:
            pf.plot_barplot_counts(
                df, "domain", env, rot=45 if env == "environment" else 0,
                savefilename=SAVE_DIR / f"domain_by_{env}.png"
            )

        # COG category abundances
        cog_abundance = [cog for cog in ["COG_S_abundance", "COG_Q_abundance"] if cog in df.columns]
        for cog in cog_abundance:
            for env in ENV_TYPES:
                pf.plot_bioindex(
                    df, env, cog,
                    savefilename=SAVE_DIR / f"{cog}_by_{env}.png"
                )

        print("Finished plotting")

        utils.print_section_separator()