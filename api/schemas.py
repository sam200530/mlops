"""Pydantic request/response models.

Design decision on the input contract: the trained model reads ~430 raw columns,
and requiring all of them would make the API unusable. So the schema names the
fields that carry most of the signal and are realistically available at
authorisation time, and accepts the long tail (``C*``, ``D*``, ``M*``, ``V*``,
``id_*``) through a single ``extra_features`` map.

Anything the caller omits becomes NaN. That is a genuine capability rather than a
shortcut: the model is a LightGBM trained on data that is 43% missing across the
V block, so it routes missing values natively and degrades gracefully instead of
failing. The response reports how many features were actually supplied, so a
caller can see the completeness of the input that produced the score.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RiskLevel = Literal["low", "medium", "high"]


class TransactionRequest(BaseModel):
    """A single transaction to score."""

    model_config = ConfigDict(extra="forbid")

    transaction_amt: Annotated[float, Field(gt=0, le=1_000_000, description="Transaction amount.")]
    product_cd: Annotated[str, Field(min_length=1, max_length=8, description="Product code.")]

    card1: Annotated[int | None, Field(default=None, ge=0, le=100_000)]
    card2: Annotated[float | None, Field(default=None, ge=0, le=10_000)]
    card3: Annotated[float | None, Field(default=None, ge=0, le=10_000)]
    card4: Annotated[str | None, Field(default=None, max_length=32)]
    card5: Annotated[float | None, Field(default=None, ge=0, le=10_000)]
    card6: Annotated[str | None, Field(default=None, max_length=32)]

    addr1: Annotated[float | None, Field(default=None, ge=0, le=10_000)]
    addr2: Annotated[float | None, Field(default=None, ge=0, le=10_000)]
    dist1: Annotated[float | None, Field(default=None, ge=0)]
    dist2: Annotated[float | None, Field(default=None, ge=0)]

    p_emaildomain: Annotated[str | None, Field(default=None, max_length=64)]
    r_emaildomain: Annotated[str | None, Field(default=None, max_length=64)]

    device_type: Annotated[str | None, Field(default=None, max_length=32)]
    device_info: Annotated[str | None, Field(default=None, max_length=128)]

    transaction_dt: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description=(
                "Seconds offset in the dataset's time base. Used for cyclical time "
                "features and velocity windows. Defaults to the model's training cut."
            ),
        ),
    ]

    extra_features: Annotated[
        dict[str, float | str | None],
        Field(
            default_factory=dict,
            description="Remaining dataset columns (C*, D*, M*, V*, id_*). Omitted -> NaN.",
        ),
    ]

    @field_validator("extra_features")
    @classmethod
    def _limit_extra_features(
        cls, value: dict[str, float | str | None]
    ) -> dict[str, float | str | None]:
        """Bound the extra map so a request cannot carry unbounded keys."""
        if len(value) > 600:
            raise ValueError("extra_features may not contain more than 600 keys")
        return value

    def to_raw_record(self) -> dict[str, Any]:
        """Flatten into dataset column names."""
        record: dict[str, Any] = {
            "TransactionAmt": self.transaction_amt,
            "ProductCD": self.product_cd,
            "card1": self.card1,
            "card2": self.card2,
            "card3": self.card3,
            "card4": self.card4,
            "card5": self.card5,
            "card6": self.card6,
            "addr1": self.addr1,
            "addr2": self.addr2,
            "dist1": self.dist1,
            "dist2": self.dist2,
            "P_emaildomain": self.p_emaildomain,
            "R_emaildomain": self.r_emaildomain,
            "DeviceType": self.device_type,
            "DeviceInfo": self.device_info,
        }
        if self.transaction_dt is not None:
            record["TransactionDT"] = self.transaction_dt
        record.update(self.extra_features)
        return record


class BatchPredictionRequest(BaseModel):
    """A batch of transactions."""

    model_config = ConfigDict(extra="forbid")

    transactions: Annotated[list[TransactionRequest], Field(min_length=1, max_length=500)]


class PredictionResponse(BaseModel):
    """Scoring result for one transaction."""

    fraud_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    risk_level: RiskLevel
    model_version: str
    request_id: str
    decision_threshold: float
    flagged: bool = Field(description="Whether the probability meets the decision threshold.")
    latency_ms: float
    features_supplied: int = Field(description="Non-null raw features provided by the caller.")


class BatchPredictionResponse(BaseModel):
    """Scoring results for a batch."""

    predictions: list[PredictionResponse]
    count: int
    model_version: str
    latency_ms: float


class FeatureContributionResponse(BaseModel):
    """One SHAP contribution."""

    feature: str
    value: float | str | None
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class ExplanationResponse(BaseModel):
    """Prediction plus the factors that drove it."""

    fraud_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    risk_level: RiskLevel
    model_version: str
    request_id: str
    base_value: float = Field(description="Model output for an average input, in log-odds.")
    top_factors: list[FeatureContributionResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    """Liveness and dependency status."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    redis: Literal["up", "down", "disabled"]
    database: Literal["up", "down", "disabled"]
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Provenance and measured performance of the loaded model."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    trained_at: str
    n_features: int
    n_train_rows: int
    calibrated: bool
    decision_threshold: float
    risk_thresholds: dict[str, float]
    validation_metrics: dict[str, Any]
    holdout_metrics: dict[str, Any]
    feature_config: dict[str, Any]
    hyperparameters: dict[str, Any]
    library_versions: dict[str, str]


class ErrorResponse(BaseModel):
    """Structured error body."""

    detail: str
    error_type: str
    request_id: str | None = None
