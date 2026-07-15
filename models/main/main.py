"""
Entry point for the econiches annotation-prediction pipeline.

Usage
-----
    python train.py --env <environment> --mod <lr|rf> --fam <cog|ec|pfam|cazy|ko|all> --root <root_dir>

The script initializes the run context, builds a model config appropriate
for the chosen estimator, runs the full experiment, and persists the models.
"""

import yaml
from joblib import dump

try:
    from econiches.modeling.init import initialize
    from econiches.modeling.run_pipeline import run_experiment
except ImportError:
    raise ImportError("Run inside econiches environment")


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

def build_lr_config(ctx) -> dict:
    """Elastic-net logistic regression config (SGDClassifier with log loss).

    The parameter grid searches over regularisation strength (alpha) and the
    L1/L2 mixing ratio (l1_ratio).  Higher alpha → stronger regularisation;
    l1_ratio=1.0 is pure L1 (sparse), l1_ratio=0.0 is pure L2 (dense).
    Fallback params are used when grid search is bypassed.
    """

    """
    "PARAM_GRID": {
        "model_type": [ctx.args.mod],
        "alpha":      [0.00001, 0.0001, 0.001, 0.01, 0.025, 0.05, 0.1],
        "l1_ratio":   [0.2, 0.3, 0.5, 0.7, 0.8],
    },
    """
    return {
        "PARAM_GRID": {
            "model_type": [ctx.args.mod],
            "alpha":      [0.001],
            "l1_ratio":   [0.2],
        },
        "FIXED_MODEL_PARAMS": dict(
            loss                = "log_loss",
            penalty             = "elasticnet",
            class_weight        = None,
            max_iter            = 3000,
            random_state        = 420,
            learning_rate       = "optimal",
            early_stopping      = True,
            validation_fraction = 0.1,
            n_iter_no_change    = 15,
            tol                 = 1e-3,
        ),
        # Fallbacks used when grid search is bypassed
        "CV_MODEL_PARAMS": dict(
            loss="log_loss", penalty="elasticnet", l1_ratio=0.5, alpha=0.001,
            class_weight="balanced", max_iter=2000, random_state=420,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, tol=1e-3,
        ),
        "FULL_MODEL_PARAMS": dict(
            loss="log_loss", penalty="elasticnet", l1_ratio=0.5, alpha=0.001,
            class_weight="balanced", max_iter=2000, random_state=420,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, tol=1e-3,
        ),
    }


def build_rf_config(ctx) -> dict:
    """Random Forest config.

    The grid searches over tree count, depth, leaf size, feature sampling
    strategy, and cost-complexity pruning (ccp_alpha).  Higher ccp_alpha
    prunes more aggressively, reducing overfitting at the cost of expressivity.
    """
    return {
        "PARAM_GRID": {
            "model_type":        [ctx.args.mod],
            "n_estimators":      [100, 200, 300],
            "max_depth":         [5, 10],
            "min_samples_leaf":  [5, 7, 8],
            "max_features":      ["sqrt", "log2"],
            "ccp_alpha":         [0.0, 0.2, 0.5, 0.7, 0.9],
        },
        "FIXED_MODEL_PARAMS": dict(
            n_jobs                 = ctx.config["n_jobs"],
            bootstrap              = True,
            class_weight           = None,
            max_samples            = 0.7,
            min_impurity_decrease  = 0.0,
            random_state           = ctx.config["seed"],
        ),
        # Fallbacks used when grid search is bypassed
        "CV_MODEL_PARAMS": dict(
            n_estimators=100, max_features="sqrt", ccp_alpha=0.0,
            n_jobs=ctx.config["n_jobs"], bootstrap=False, class_weight=None,
            min_impurity_decrease=0.0, random_state=ctx.config["seed"],
        ),
        "FULL_MODEL_PARAMS": dict(
            n_estimators=100, max_features="sqrt", ccp_alpha=0.0,
            n_jobs=ctx.config["n_jobs"], bootstrap=False, class_weight=None,
            min_impurity_decrease=0.0, random_state=ctx.config["seed"],
        ),
    }


_CONFIG_BUILDERS = {
    "lr": build_lr_config,
    "rf": build_rf_config,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ctx = initialize(__file__)

    # Select and build the model config for the requested estimator
    build_config = _CONFIG_BUILDERS.get(ctx.args.mod)
    if build_config is None:
        raise ValueError(f"Unknown model type '{ctx.args.mod}'. Choose from: {list(_CONFIG_BUILDERS)}")
    config = build_config(ctx)

    # Persist the config used for this run so results are fully reproducible
    config_out_path = ctx.paths.logs / "config.yaml"
    ctx.paths.logs.mkdir(parents=True, exist_ok=True)
    with open(config_out_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    results = run_experiment(ctx=ctx, config=config, sparse=True)

    # Save the full-refit models (one classifier per annotation class)
    model_path = ctx.paths.logs / "model.joblib"
    dump(results["full_models"], model_path, compress=("zlib", 3))
    ctx.log.info("Saved full-fit models to: %s", model_path)