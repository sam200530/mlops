"""Data-quality checks for incoming traffic.

Distinct from drift: drift asks "has the distribution moved", data quality asks
"is this input well-formed at all". The two fail differently and need different
responses — drift means retrain, quality means fix the caller.

The reference profile is built from the training data at training time and shipped
alongside the model, so checks compare live traffic against the exact distribution
the model learned from.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReferenceProfile:
    """Per-column expectations captured from the training data."""

    missing_rates: dict[str, float] = field(default_factory=dict)
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    categorical_values: dict[str, list[str]] = field(default_factory=dict)
    n_rows: int = 0

    @classmethod
    def from_frame(cls, df: pd.DataFrame, max_categories: int = 200) -> ReferenceProfile:
        """Build a profile from a training frame."""
        profile = cls(n_rows=len(df))
        for column in df.columns:
            series = df[column]
            profile.missing_rates[column] = float(series.isna().mean())
            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if not non_null.empty:
                    profile.numeric_ranges[column] = (
                        float(non_null.min()),
                        float(non_null.max()),
                    )
            else:
                values = series.dropna().astype(str).value_counts().head(max_categories)
                profile.categorical_values[column] = values.index.tolist()
        return profile

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "n_rows": self.n_rows,
                    "missing_rates": self.missing_rates,
                    "numeric_ranges": {k: list(v) for k, v in self.numeric_ranges.items()},
                    "categorical_values": self.categorical_values,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote reference profile to %s (%d columns)", path, len(self.missing_rates))
        return path

    @classmethod
    def load(cls, path: Path) -> ReferenceProfile:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            missing_rates=payload.get("missing_rates", {}),
            numeric_ranges={
                k: (float(v[0]), float(v[1])) for k, v in payload.get("numeric_ranges", {}).items()
            },
            categorical_values=payload.get("categorical_values", {}),
            n_rows=int(payload.get("n_rows", 0)),
        )


@dataclass
class QualityIssue:
    """One detected data-quality problem."""

    column: str
    issue: str
    detail: str
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "issue": self.issue,
            "detail": self.detail,
            "severity": self.severity,
        }


def check_quality(
    df: pd.DataFrame,
    profile: ReferenceProfile,
    missing_rate_tolerance: float = 0.20,
    out_of_range_tolerance: float = 0.01,
) -> list[QualityIssue]:
    """Compare a batch against the reference profile.

    Args:
        df: Incoming data.
        profile: Training-time reference.
        missing_rate_tolerance: Absolute increase in missing rate that counts as
            an issue. Absolute rather than relative because a column going from
            0% to 15% missing matters, while 90% to 92% does not.
        out_of_range_tolerance: Fraction of values allowed outside the training
            range before flagging. Not zero, because a genuinely larger
            transaction than any seen in training is plausible, not a bug.

    Returns:
        Detected issues, most severe first.
    """
    issues: list[QualityIssue] = []

    expected = set(profile.missing_rates)
    actual = set(df.columns)
    for column in sorted(expected - actual):
        issues.append(
            QualityIssue(column, "missing_column", "Column absent from incoming data", "high")
        )
    for column in sorted(actual - expected):
        issues.append(
            QualityIssue(column, "unexpected_column", "Column not present in training", "medium")
        )

    for column in sorted(expected & actual):
        series = df[column]
        observed_missing = float(series.isna().mean())
        reference_missing = profile.missing_rates[column]
        if observed_missing - reference_missing > missing_rate_tolerance:
            issues.append(
                QualityIssue(
                    column,
                    "missing_rate_increase",
                    f"{reference_missing:.2%} -> {observed_missing:.2%}",
                    "high" if observed_missing > 0.5 else "medium",
                )
            )

        if column in profile.numeric_ranges and pd.api.types.is_numeric_dtype(series):
            low, high = profile.numeric_ranges[column]
            non_null = series.dropna()
            if not non_null.empty:
                outside = float(((non_null < low) | (non_null > high)).mean())
                if outside > out_of_range_tolerance:
                    issues.append(
                        QualityIssue(
                            column,
                            "out_of_range",
                            f"{outside:.2%} outside training range [{low:g}, {high:g}]",
                            "medium",
                        )
                    )

        if column in profile.categorical_values:
            known = set(profile.categorical_values[column])
            if known:
                values = series.dropna().astype(str)
                if not values.empty:
                    unseen = float((~values.isin(known)).mean())
                    if unseen > 0.05:
                        issues.append(
                            QualityIssue(
                                column,
                                "unseen_categories",
                                f"{unseen:.2%} of values unseen in training",
                                "medium",
                            )
                        )

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(issues, key=lambda issue: order.get(issue.severity, 3))


def summarise_issues(issues: list[QualityIssue]) -> dict[str, Any]:
    """Compact summary for a monitoring report."""
    return {
        "n_issues": len(issues),
        "high": sum(1 for i in issues if i.severity == "high"),
        "medium": sum(1 for i in issues if i.severity == "medium"),
        "issues": [i.to_dict() for i in issues[:50]],
    }
