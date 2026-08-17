"""Tests for drift detection, data quality and the metrics store."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.data_quality import ReferenceProfile, check_quality, summarise_issues
from src.monitoring.drift import (
    classify_psi,
    feature_drift,
    population_stability_index,
    prediction_drift,
    summarise,
)
from src.monitoring.metrics_store import MetricsStore


class TestPSI:
    def test_identical_distributions_give_near_zero_psi(self) -> None:
        generator = np.random.default_rng(5)
        sample = generator.normal(size=5000)
        other = generator.normal(size=5000)
        assert population_stability_index(sample, other) < 0.1

    def test_shifted_distribution_is_detected(self) -> None:
        generator = np.random.default_rng(6)
        reference = generator.normal(0, 1, size=5000)
        current = generator.normal(3, 1, size=5000)
        psi = population_stability_index(reference, current)
        assert psi > 0.25
        assert classify_psi(psi) == "significant"

    def test_psi_is_finite_when_a_bin_is_empty(self) -> None:
        reference = np.concatenate([np.zeros(1000), np.ones(1000)])
        current = np.zeros(1000)
        assert np.isfinite(population_stability_index(reference, current))

    def test_near_constant_feature_does_not_explode(self) -> None:
        assert np.isfinite(population_stability_index(np.ones(100), np.ones(100)))

    def test_empty_input_returns_nan(self) -> None:
        assert np.isnan(population_stability_index(np.array([]), np.array([1.0])))

    def test_thresholds_map_to_verdicts(self) -> None:
        assert classify_psi(0.05) == "stable"
        assert classify_psi(0.15) == "moderate"
        assert classify_psi(0.40) == "significant"


class TestFeatureDrift:
    def test_detects_drift_in_one_column_only(self) -> None:
        generator = np.random.default_rng(7)
        reference = pd.DataFrame(
            {
                "stable": generator.normal(size=3000),
                "drifting": generator.normal(size=3000),
                "category": generator.choice(["a", "b"], size=3000),
            }
        )
        current = pd.DataFrame(
            {
                "stable": generator.normal(size=3000),
                "drifting": generator.normal(5, 1, size=3000),
                "category": generator.choice(["a", "b"], size=3000),
            }
        )
        table = feature_drift(reference, current)
        assert table.iloc[0]["feature"] == "drifting"
        assert table.iloc[0]["verdict"] == "significant"
        stable_row = table[table["feature"] == "stable"].iloc[0]
        assert stable_row["verdict"] == "stable"

    def test_reports_missing_rate_change(self) -> None:
        reference = pd.DataFrame({"x": [1.0] * 100})
        current = pd.DataFrame({"x": [1.0] * 50 + [np.nan] * 50})
        table = feature_drift(reference, current)
        assert table.iloc[0]["current_missing_rate"] == pytest.approx(0.5)
        assert table.iloc[0]["missing_rate_delta"] == pytest.approx(0.5)

    def test_summarise_counts_verdicts(self) -> None:
        generator = np.random.default_rng(8)
        reference = pd.DataFrame({"a": generator.normal(size=500)})
        current = pd.DataFrame({"a": generator.normal(size=500)})
        summary = summarise(feature_drift(reference, current))
        assert summary["n_features"] == 1
        assert summary["stable"] + summary["moderate"] + summary["significant"] == 1


class TestPredictionDrift:
    def test_detects_a_score_distribution_shift(self) -> None:
        generator = np.random.default_rng(9)
        reference = generator.beta(1, 20, size=3000)
        current = generator.beta(3, 5, size=3000)
        report = prediction_drift(reference, current)
        assert report["verdict"] == "significant"
        assert report["current_mean"] > report["reference_mean"]


class TestDataQuality:
    def test_flags_missing_and_unexpected_columns(self) -> None:
        reference = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        profile = ReferenceProfile.from_frame(reference)
        current = pd.DataFrame({"a": [1.0], "c": [5.0]})
        issues = check_quality(current, profile)
        kinds = {issue.issue for issue in issues}
        assert "missing_column" in kinds
        assert "unexpected_column" in kinds

    def test_flags_missing_rate_increase(self) -> None:
        profile = ReferenceProfile.from_frame(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}))
        current = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan]})
        issues = check_quality(current, profile)
        assert any(issue.issue == "missing_rate_increase" for issue in issues)

    def test_flags_out_of_range_values(self) -> None:
        profile = ReferenceProfile.from_frame(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
        current = pd.DataFrame({"a": [500.0] * 10})
        issues = check_quality(current, profile)
        assert any(issue.issue == "out_of_range" for issue in issues)

    def test_flags_unseen_categories(self) -> None:
        profile = ReferenceProfile.from_frame(pd.DataFrame({"c": ["a", "b", "a"]}))
        current = pd.DataFrame({"c": ["zzz"] * 10})
        issues = check_quality(current, profile)
        assert any(issue.issue == "unseen_categories" for issue in issues)

    def test_clean_data_produces_no_issues(self) -> None:
        reference = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "c": ["x", "y", "x", "y"]})
        profile = ReferenceProfile.from_frame(reference)
        assert check_quality(reference.copy(), profile) == []

    def test_profile_roundtrip(self, tmp_path) -> None:
        reference = pd.DataFrame({"a": [1.0, 2.0], "c": ["x", "y"]})
        profile = ReferenceProfile.from_frame(reference)
        path = profile.save(tmp_path / "profile.json")
        loaded = ReferenceProfile.load(path)
        assert loaded.missing_rates == profile.missing_rates
        assert loaded.numeric_ranges == profile.numeric_ranges

    def test_summarise_issues_counts_severities(self) -> None:
        profile = ReferenceProfile.from_frame(pd.DataFrame({"a": [1.0, 2.0]}))
        summary = summarise_issues(check_quality(pd.DataFrame({"b": [1.0]}), profile))
        assert summary["n_issues"] >= 1


class TestMetricsStore:
    def test_counters_and_reservoirs_accumulate(self) -> None:
        store = MetricsStore(None)
        store.increment("requests", 3)
        store.increment("requests")
        store.observe_latency(10.0)
        store.observe_latency(20.0)
        store.observe_score(0.9)

        assert store.counters()["requests"] == 4
        latency = store.latency_summary()
        assert latency["count"] == 2
        assert latency["mean"] == pytest.approx(15.0)
        assert store.score_summary()["count"] == 1
        assert store.backend == "in_memory_fallback"

    def test_empty_summaries_are_nan_not_errors(self) -> None:
        summary = MetricsStore(None).latency_summary()
        assert summary["count"] == 0
        assert np.isnan(summary["mean"])

    def test_reset_clears_state(self) -> None:
        store = MetricsStore(None)
        store.increment("x")
        store.reset()
        assert store.counters() == {}
