"""Tests for the estimator factories and the shared early-stopping path.

Both boosted models are exercised, because the project compares them and a
comparison is only meaningful if each is configured the way it is claimed to be.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.estimators import (
    BOOSTED_MODELS,
    REQUIRES_DENSE_IMPUTED,
    build_model,
    fit_with_early_stopping,
    scale_pos_weight,
)


class TestScalePosWeight:
    def test_is_the_negative_to_positive_ratio(self) -> None:
        y = np.array([0] * 90 + [1] * 10)
        assert scale_pos_weight(y) == pytest.approx(9.0)

    def test_rejects_a_single_class(self) -> None:
        with pytest.raises(ValueError, match="No positive samples"):
            scale_pos_weight(np.zeros(10))


class TestBuildModel:
    @pytest.mark.parametrize(
        "name", ["logistic_regression", "random_forest", "lightgbm", "xgboost"]
    )
    def test_every_declared_model_builds(self, name) -> None:
        assert build_model(name, seed=1, n_jobs=1) is not None

    def test_unknown_model_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("catboost", seed=1)  # type: ignore[arg-type]

    def test_boosters_take_native_categoricals_and_nan(self) -> None:
        """Neither booster may be handed an imputed dense matrix — routing NaN
        natively is why they are preferred on this dataset."""
        assert {"lightgbm", "xgboost"} == BOOSTED_MODELS
        assert not (BOOSTED_MODELS & REQUIRES_DENSE_IMPUTED)

    def test_xgboost_is_configured_for_categoricals_and_pr_auc(self) -> None:
        params = build_model("xgboost", seed=1, n_jobs=1).get_params()
        assert params["enable_categorical"] is True
        assert params["tree_method"] == "hist"
        # PR-AUC is the project's selection metric; early stopping must optimise it.
        assert params["eval_metric"] == "aucpr"

    def test_imbalance_weight_reaches_both_boosters(self) -> None:
        lgbm = build_model("lightgbm", seed=1, n_jobs=1, imbalance_weight=27.6).get_params()
        xgboost = build_model("xgboost", seed=1, n_jobs=1, imbalance_weight=27.6).get_params()
        assert lgbm["scale_pos_weight"] == pytest.approx(27.6)
        assert xgboost["scale_pos_weight"] == pytest.approx(27.6)

    def test_overrides_win_over_defaults(self) -> None:
        assert build_model("xgboost", seed=1, max_depth=3).get_params()["max_depth"] == 3


class TestEarlyStopping:
    @pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
    def test_both_boosters_fit_and_report_an_iteration(self, name, prepared_frame) -> None:
        """The two libraries express early stopping differently; the helper hides
        that, and both must return a usable best-iteration count."""
        from src.features.pipeline import FeaturePipeline

        cut = int(len(prepared_frame) * 0.7)
        pipeline = FeaturePipeline()
        pipeline.fit(prepared_frame.iloc[:cut])
        X = pipeline.transform(prepared_frame.iloc[:cut])
        y = prepared_frame.iloc[:cut]["isFraud"].to_numpy()
        split = int(len(X) * 0.8)

        model = build_model(name, seed=1, n_jobs=1, n_estimators=30)
        best = fit_with_early_stopping(
            model,
            name,
            X.iloc[:split],
            y[:split],
            X.iloc[split:],
            y[split:],
            pipeline.categorical_features,
        )
        assert isinstance(best, int) and best >= 0
        proba = model.predict_proba(X.iloc[split:])[:, 1]
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_non_boosted_model_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not support early stopping"):
            fit_with_early_stopping(None, "random_forest", None, None, None, None, [])
