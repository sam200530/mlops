"""Model factories.

Three families, each with a distinct job in the comparison:

* **Logistic Regression** — the floor. A regularized linear model on imputed,
  scaled, one-hot features. If a complex model cannot clearly beat this, the
  complexity is not earning its keep. It also reveals how much of the signal is
  simply linear.
* **Random Forest** — the bagging contrast. Non-linear and interaction-capable
  but, unlike LightGBM, it cannot route NaN structurally, so it must be given
  imputed inputs. That difference is itself informative about how much of
  LightGBM's advantage comes from native missing-value handling.
* **LightGBM** — the candidate for production. Native categorical support,
  native NaN routing, and the only one of the three that is fast enough at
  590,540 × ~550 to tune properly.

Imbalance is handled by **reweighting, not resampling**. SMOTE would interpolate
between fraud rows across a ~550-column space that is largely categorical and
43% missing in the V block; the interpolants would not be plausible
transactions. ``scale_pos_weight`` / ``class_weight`` leaves the data honest and
only changes the loss. Because reweighting distorts predicted probabilities, and
this project's API returns a probability that drives a risk band, calibration is
applied afterwards rather than treated as optional.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

ModelName = Literal["logistic_regression", "random_forest", "lightgbm"]

#: Models that cannot accept NaN and therefore require the imputed matrix.
REQUIRES_DENSE_IMPUTED: frozenset[str] = frozenset({"logistic_regression", "random_forest"})


def scale_pos_weight(y: np.ndarray) -> float:
    """Ratio of negatives to positives, for LightGBM's ``scale_pos_weight``."""
    positives = float(np.sum(y))
    if positives == 0:
        raise ValueError("No positive samples — cannot compute scale_pos_weight")
    return float((len(y) - positives) / positives)


def lightgbm_params(
    seed: int = 42,
    n_jobs: int = -1,
    imbalance_weight: float | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Baseline LightGBM parameters.

    Defaults chosen for this dataset rather than copied from a tutorial:

    * ``num_leaves`` 64 with ``min_child_samples`` 100 — at 3.5% prevalence a
      leaf needs enough rows to contain a meaningful number of positives, or the
      model fits noise in the tail.
    * ``feature_fraction`` 0.6 — 339 of the ~550 features are the highly
      redundant V block; subsampling columns decorrelates trees cheaply, which
      is the intended substitute for the correlation-clustering we chose not to
      do before establishing a baseline.
    * ``max_bin`` 127 rather than the default 255 — halves the histogram memory
      on a wide frame with negligible accuracy cost.
    """
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "n_estimators": 2000,
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbose": -1,
    }
    if imbalance_weight is not None:
        params["scale_pos_weight"] = imbalance_weight
    params.update(overrides)
    return params


def make_lightgbm(
    seed: int = 42,
    n_jobs: int = -1,
    imbalance_weight: float | None = None,
    **overrides: Any,
) -> lgb.LGBMClassifier:
    """Build a LightGBM classifier."""
    return lgb.LGBMClassifier(
        **lightgbm_params(seed=seed, n_jobs=n_jobs, imbalance_weight=imbalance_weight, **overrides)
    )


def make_random_forest(
    seed: int = 42, n_jobs: int = -1, **overrides: Any
) -> RandomForestClassifier:
    """Build a Random Forest classifier.

    ``max_depth`` and ``min_samples_leaf`` are capped deliberately: unbounded
    trees on 470k rows produce a model of several GB, and this machine has
    single-digit GB of headroom. The cap is a documented constraint, not a
    tuning result.
    """
    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 16,
        "min_samples_leaf": 50,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbose": 0,
    }
    params.update(overrides)
    return RandomForestClassifier(**params)


def make_logistic_regression(seed: int = 42, **overrides: Any) -> LogisticRegression:
    """Build a regularized logistic regression.

    ``saga`` is avoided (too slow at this scale) in favour of ``lbfgs``, which
    handles the dense standardised matrix well. ``class_weight='balanced'``
    matches the reweighting strategy used for the tree models so the comparison
    isolates model capacity rather than imbalance handling.
    """
    params: dict[str, Any] = {
        "penalty": "l2",
        "C": 0.1,
        "solver": "lbfgs",
        "max_iter": 300,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": None,
    }
    params.update(overrides)
    return LogisticRegression(**params)


def build_model(
    name: ModelName,
    seed: int = 42,
    n_jobs: int = -1,
    imbalance_weight: float | None = None,
    **overrides: Any,
):
    """Dispatch to the requested model factory."""
    if name == "lightgbm":
        return make_lightgbm(
            seed=seed, n_jobs=n_jobs, imbalance_weight=imbalance_weight, **overrides
        )
    if name == "random_forest":
        return make_random_forest(seed=seed, n_jobs=n_jobs, **overrides)
    if name == "logistic_regression":
        return make_logistic_regression(seed=seed, **overrides)
    raise ValueError(f"Unknown model {name!r}")
