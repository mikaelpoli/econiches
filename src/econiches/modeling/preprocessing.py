import warnings

import pandas as pd

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*`sklearn\.utils\.parallel\.delayed`.*"
)

from sklearn.preprocessing import MultiLabelBinarizer


# =========================================================
# FEATURE SELECTION
# =========================================================
def select_annotation_family(
    df: pd.DataFrame,
    family: str,
    id_col: str = "mag_id"
) -> pd.DataFrame:
    """
    Select feature columns by annotation family.
    Always returns a DataFrame including id_col.
    """

    match family:
        case "none":
            cols = []
        case "all":
            cols = df.select_dtypes(include=["number"]).columns.tolist()
        case "cog":
            cols = [c for c in df.columns if c.startswith("COG")]
        case "ec":
            cols = [c for c in df.columns if c.startswith("EC")]
        case "pfam":
            cols = [c for c in df.columns if c.startswith("PFAM")]
        case "cazy":
            cols = [c for c in df.columns if c.startswith("CAZy")]
        case "ko":
            cols = [c for c in df.columns if c.startswith("KO")]

    print(f"Selected {len(cols)} `{family.upper()}` features.")

    return df[[id_col] + cols].copy()


# =========================================================
# CORE PREPROCESSING (TRAIN)
# =========================================================
def prepare_multilabel_data(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    label_col: str,
    group_col: str = "metagenome_id",
    id_col: str = "mag_id",
    aggregation: str = "unique",
):
    """
    Full pipeline:
    - aggregate X (one row per MAG)
    - build multilabel targets
    - align X, Y, and groups
    """

    # -------------------------
    # 1. CLEAN LABEL DATA
    # -------------------------
    y_df = y_df[[id_col, label_col, group_col]].drop_duplicates()

    # -------------------------
    # 2. MULTILABEL TARGETS
    # -------------------------
    grouped_labels = (
        y_df
        .groupby(id_col)[label_col]
        .apply(lambda x: sorted(x.unique()))
    )

    # -------------------------
    # 3. GROUP VECTOR (MAG → GROUP)
    # -------------------------
    mag_to_group = (
        y_df[[id_col, group_col]]
        .drop_duplicates()
        .set_index(id_col)[group_col]
    )

    # Ensure 1 group per MAG
    group_counts = y_df.groupby(id_col)[group_col].nunique()
    assert (group_counts == 1).all(), "A MAG belongs to multiple groups!"

    # -------------------------
    # 4. AGGREGATE FEATURES
    # -------------------------
    if aggregation == "unique":
        X_agg = X_df.groupby(id_col).first().reset_index()
    else:
        X_agg = X_df

    # -------------------------
    # 5. ALIGN EVERYTHING
    # -------------------------
    X_agg = X_agg.sort_values(id_col)
    ids = X_agg[id_col].values

    grouped_labels = grouped_labels.reindex(ids)
    mag_to_group = mag_to_group.reindex(ids)

    assert not grouped_labels.isnull().any(), "Missing labels after reindex!"
    assert not mag_to_group.isnull().any(), "Missing groups after reindex!"

    # -------------------------
    # 6. BINARIZE LABELS
    # -------------------------
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(grouped_labels)

    # -------------------------
    # 7. BUILD MATRICES
    # -------------------------
    feature_names = X_agg.columns.tolist()

    X = X_agg.drop(columns=[id_col]).values
    groups = mag_to_group.values

    assert len(X) == len(Y) == len(groups)

    return X, Y, groups, ids, mlb, feature_names


# =========================================================
# TEST SET TRANSFORM (USES TRAINED MLB)
# =========================================================
def transform_multilabel_test(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    mlb: MultiLabelBinarizer,
    label_col: str,
    group_col: str = "metagenome_id",
    id_col: str = "mag_id",
    aggregation: str = "unique",
):
    """
    Same pipeline as training, but:
    - uses pre-fitted MultiLabelBinarizer
    """

    y_df = y_df[[id_col, label_col, group_col]].drop_duplicates()

    grouped_labels = (
        y_df
        .groupby(id_col)[label_col]
        .apply(lambda x: sorted(x.unique()))
    )

    mag_to_group = (
        y_df[[id_col, group_col]]
        .drop_duplicates()
        .set_index(id_col)[group_col]
    )

    # Aggregate X
    if aggregation == "unique":
        X_agg = X_df.groupby(id_col).first().reset_index()
    else:
        X_agg = X_df

    # Align
    X_agg = X_agg.sort_values(id_col)
    ids = X_agg[id_col].values

    grouped_labels = grouped_labels.reindex(ids)
    mag_to_group = mag_to_group.reindex(ids)

    assert not grouped_labels.isnull().any()
    assert not mag_to_group.isnull().any()

    # Transform labels using TRAIN mlb
    Y = mlb.transform(grouped_labels)

    feature_names = X_agg.columns.tolist()

    X = X_agg.drop(columns=[id_col]).values
    groups = mag_to_group.values

    assert len(X) == len(Y) == len(groups)

    return X, Y, groups, ids, feature_names


# =========================================================
# HIGH-LEVEL HELPER
# =========================================================
def build_train_test(
    X_train_df,
    X_test_df,
    y_train_df,
    y_test_df,
    family,
    label_col,
    group_col="metagenome_id",
    id_col="mag_id",
    aggregation="unique",
    mode="mlb"
):
    """
    End-to-end helper:
    - feature selection
    - preprocessing
    """
    if mode not in ["mlb", "mc"]:
        raise ValueError("Choose desidred output: multilabel ('mlb') or multiclass ('mc')")
    
    def _build_dict(dataset_types):
        return {dt: None for dt in dataset_types}

    dataset_types = ["train", "test"]

    features_dfs = {
        "train": X_train_df,
        "test": X_test_df
    }
    labels_dfs = {
        "train": y_train_df,
        "test": y_test_df
    }

    features = _build_dict(dataset_types)
    labels = _build_dict(dataset_types)
    ids = _build_dict(dataset_types)
    groups = _build_dict(dataset_types)
    feature_names = _build_dict(dataset_types)
    
    mlb = None

    print("Selecting features...")
    
    for dt in dataset_types:
        print(f"=> {dt.title()} set")
        features_dfs[dt] = select_annotation_family(features_dfs[dt], family, id_col)

        if mode == "mc":
            features[dt] = (
                features_dfs[dt].drop_duplicates(subset=id_col)
                .sort_values(id_col)
            )

            ids[dt] = features[dt][id_col].values

            groups[dt] = (
                labels_dfs[dt]
                .groupby(id_col)[label_col]
                .apply(lambda x: sorted(x.unique()))
                .reindex(ids[dt])
            )

            if groups[dt].isnull().any():
                raise ValueError(f"Missing labels in y_{dt} after reindex")
            
            if dt == "train":
                mlb = MultiLabelBinarizer()
                labels[dt] = mlb.fit_transform(groups[dt])

            if dt == "test":
                labels[dt] = mlb.transform(groups[dt])

        elif mode=="mlb":

            if dt == "train":
                features[dt], labels[dt], groups[dt], ids[dt], mlb, feature_names[dt] = prepare_multilabel_data(
                    features_dfs[dt],
                    labels_dfs[dt],
                    label_col,
                    group_col,
                    id_col,
                    aggregation
                )
            
            if dt == "test":
                features[dt], labels[dt], groups[dt], ids[dt], feature_names[dt] = transform_multilabel_test(
                    features_dfs[dt],
                    labels_dfs[dt],
                    mlb,
                    label_col,
                    group_col,
                    id_col,
                    aggregation
                )

    return {
        "X_train": features["train"],
        "y_train": labels["train"],
        "groups_train": groups["train"],
        "ids_train": ids["train"],
        "X_test": features["test"],
        "y_test": labels["test"],
        "groups_test": groups["test"],
        "ids_test": ids["test"],
        "mlb": mlb,
        "feature_names_train": [x for x in feature_names["train"] if x not in [id_col, label_col]],
        "feature_names_test": [x for x in feature_names["test"] if x not in [id_col, label_col]]
    }