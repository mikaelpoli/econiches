"""
Initialization and preprocessing utilities for the econiches ML pipeline.

Typical usage
-------------
ctx = initialize(__file__)          # in any pipeline script
X_tr, X_te = transform_data(        # optional feature scaling
    ctx.data["X_train"], ctx.data["X_test"],
    log=ctx.log, scaler="StandardScaler",
)
"""
from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace

import yaml 
import numpy as np
import pandas as pd
import sklearn.preprocessing

try:
    from econiches.modeling import paths
    from econiches.modeling import paths_val
    from econiches.modeling.logger import get_logger
    from econiches.modeling.preprocessing import build_train_test
except ImportError:
    raise ImportError("Run inside econiches environment")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAG_ID_COL        = "mag_id"
METAGENOME_ID_COL = "metagenome_id"

# Canonical feature-family ordering used when `--fam all/combined` is set.
# Deterministic column order is critical for cross-run reproducibility.
_COMBINED_FAMILY_ORDER = ["COG", "EC", "PFAM", "KO", "CAZy"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialize(caller_file: str) -> SimpleNamespace:
    """Bootstrap a pipeline run and return a context object with all shared state.

    Performs, in order:
      1. Parse CLI arguments and resolve filesystem paths.
      2. Load YAML/JSON config and infer the label encoding mode.
      3. Set up a timestamped file logger.
      4. Load raw train/test splits (features + labels) from disk.
      5. Canonicalise feature-column order for combined-family experiments.
      6. Build processed arrays / label-binarizer via ``build_train_test``.

    Parameters
    ----------
    caller_file:
        Pass `__file__` from the calling script. Its basename is embedded
        in the logger name so log messages identify their source script.

    Returns
    -------
    SimpleNamespace with attributes:
        - args             -> parsed CLI namespace
        - paths            -> resolved `paths.Paths` object
        - config           -> raw config dict
        - is_multilabel    -> True when the task is multi-label classification
        - run_id           -> `YYYYMMDD_HHMMSS` string that tags this run
        - log              -> configured `logging.Logger` instance
        - mag_id_col       -> column name for MAG identifiers
        - metagenome_id_col-> column name for metagenome identifiers
        - data             -> dict returned by `build_train_test`
        - class_names      -> ordered list of label strings
        - id_to_lab        -> `{int_index: label_string}` mapping
    """
    print("--- INITIALIZATION ---\n")

    run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.basename(caller_file)

    # ---- Paths and config ----
    args       = paths.parse_args()
    proj_paths = _build_paths(args, run_id)
    with open(proj_paths.files["config"]) as f:
        config = yaml.safe_load(f)
    
    is_multilabel = config["multilabel"]

    # ---- Logger ----
    log = get_logger(
        name=f"econiches_{script_name}",
        log_file=proj_paths.logs / f"{run_id}.log",
    )
    print(f"\nInitialized logger: {log}")

    # ---- Data loading ----
    raw = _load_raw_splits(proj_paths)

    if _is_combined_family(args.fam):
        raw = _reorder_combined_features(raw)

    processed = build_train_test(
        X_train_df  = raw["X_train"],
        X_test_df   = raw["X_test"],
        y_train_df  = raw["y_train"],
        y_test_df   = raw["y_test"],
        family      = args.fam,
        label_col   = args.env,
        group_col   = METAGENOME_ID_COL,
        id_col      = MAG_ID_COL,
        aggregation = "unique",
        mode        = "mlb" if is_multilabel else "mc",
    )

    class_names = list(processed["mlb"].classes_)

    return SimpleNamespace(
        args              = args,
        paths             = proj_paths,
        config            = config,
        is_multilabel     = is_multilabel,
        run_id            = run_id,
        log               = log,
        mag_id_col        = MAG_ID_COL,
        metagenome_id_col = METAGENOME_ID_COL,
        data              = processed,
        class_names       = class_names,
        id_to_lab         = {i: lab for i, lab in enumerate(class_names)},
    )


def transform_data(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    log,
    *,
    scaler:    str  | None = None,
    log_scale: bool        = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally scale features, fitting only on the training split.

    Transformations are applied in order: scaler first, then log1p.
    Either, both, or neither may be active.

    Parameters
    ----------
    X_train, X_test:
        Raw feature arrays.  The scaler is *fit* on `X_train` only and
        then *applied* to both, preventing data leakage.
    log:
        Logger instance (from the run context).
    scaler:
        Name of any `sklearn.preprocessing` scaler class, e.g.
        `"StandardScaler"`, `"MinMaxScaler"`.  `None` skips scaling.
    log_scale:
        When `True`, applies `numpy.log1p` element-wise. Useful for
        count data (e.g. gene-family abundances).

    Returns
    -------
    Tuple of (transformed X_train, transformed X_test).

    Raises
    ------
    ValueError
        If `scaler` is not a recognised `sklearn.preprocessing` class.
    """
    X_train_out, X_test_out = X_train, X_test

    scaler_obj = None

    if scaler is not None:
        if scaler not in sklearn.preprocessing.__all__:
            raise ValueError(
                f"Unknown scaler '{scaler}'. "
                f"Choose from: {sorted(sklearn.preprocessing.__all__)}"
            )
        scaler_obj  = getattr(sklearn.preprocessing, scaler)()
        X_train_out = scaler_obj.fit_transform(X_train_out)
        X_test_out  = scaler_obj.transform(X_test_out)
        log.info("Applied %s scaling.", scaler)

    if log_scale:
        X_train_out = np.log1p(X_train_out)
        X_test_out  = np.log1p(X_test_out)
        log.info("Applied log1p transformation.")

    return X_train_out, X_test_out, scaler_obj


# ---------------------------------------------------------------------------
# Private helpers  (underscore prefix = internal; not part of public API)
# ---------------------------------------------------------------------------

def _build_paths(args, run_id: str) -> paths.Paths:
    """Construct, validate, and display the project path tree."""
    proj_paths = paths.Paths(
        env         = args.env,
        root_dir    = args.root.resolve(),
        model_type  = args.mod,
        annot_family= args.fam,
        log_run_dir = run_id,
    )
    paths.ensure_dirs(proj_paths)
    paths.print_paths(proj_paths)
    return proj_paths


def _load_raw_splits(proj_paths: paths.Paths) -> dict[str, pd.DataFrame]:
    """Read all four train/test splits from disk, sorted by MAG ID for alignment."""
    return {
        "X_train": pd.read_parquet(proj_paths.files["X_train"]).sort_values(MAG_ID_COL),
        "y_train": pd.read_csv(proj_paths.files["y_train"]).sort_values(MAG_ID_COL),
        "X_test":  pd.read_parquet(proj_paths.files["X_test"]).sort_values(MAG_ID_COL),
        "y_test":  pd.read_csv(proj_paths.files["y_test"]).sort_values(MAG_ID_COL),
    }


def _is_combined_family(family: str) -> bool:
    """Return True when the experiment merges all annotation families."""
    return family.lower() in {"all", "combined"}


def _reorder_combined_features(
    raw: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Sort feature columns into a canonical family order for reproducibility.

    When features from multiple annotation families (COG, EC, PFAM, …) are
    concatenated, the resulting column order may vary across runs or machines.
    This function enforces `_COMBINED_FAMILY_ORDER` so that model weights
    remain comparable across experiments.
    """
    prefix_rank = {fam: rank for rank, fam in enumerate(_COMBINED_FAMILY_ORDER)}
    n_unknown   = len(_COMBINED_FAMILY_ORDER) # rank for unrecognised prefixes

    def _family_sort_key(col: str) -> tuple[int, str]:
        family = next(
            (fam for fam in _COMBINED_FAMILY_ORDER if col.startswith(f"{fam}_")),
            None,
        )
        return (prefix_rank.get(family, n_unknown), col)

    for split in ("train", "test"):
        key = f"X_{split}"
        df  = raw[key]
        raw[key] = df.reindex(columns=sorted(df.columns, key=_family_sort_key))

    return raw