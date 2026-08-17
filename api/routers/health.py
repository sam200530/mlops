"""Health and model-info endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from api.dependencies import AppState, get_state, require_artifact
from api.schemas import HealthResponse, ModelInfoResponse
from database.session import check_connection
from src.models.artifact import ModelArtifact

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    """Liveness plus dependency status.

    Reports ``degraded`` rather than failing when Redis or Postgres are down: the
    service can still score transactions without them, and a health check that
    returns 503 would take the service out of a load balancer for a
    non-critical dependency.
    """
    redis_status = (
        "disabled"
        if not state.settings.enable_redis
        else ("up" if state.redis_client is not None else "down")
    )
    if not state.settings.enable_prediction_log:
        database_status = "disabled"
    else:
        database_status = "up" if check_connection() else "down"

    model_loaded = state.artifact is not None
    degraded = not model_loaded or redis_status == "down" or database_status == "down"

    return HealthResponse(
        status="degraded" if degraded else "ok",
        model_loaded=model_loaded,
        model_version=state.model_version,
        redis=redis_status,  # type: ignore[arg-type]
        database=database_status,  # type: ignore[arg-type]
        uptime_seconds=round(state.metrics.uptime_seconds(), 1),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    artifact: ModelArtifact = Depends(require_artifact),
    state: AppState = Depends(get_state),
) -> ModelInfoResponse:
    """Provenance and measured performance of the model currently serving.

    The holdout metrics returned here are the single-evaluation numbers from
    training, not recomputed live — recomputing them would require labels the
    service does not have at inference time.
    """
    metadata = artifact.metadata
    return ModelInfoResponse(
        model_name=metadata.model_name,
        model_version=state.model_version,
        trained_at=metadata.trained_at,
        n_features=metadata.n_features,
        n_train_rows=metadata.n_train_rows,
        calibrated=metadata.calibrated,
        decision_threshold=artifact.decision_threshold,
        risk_thresholds={
            "medium": state.settings.risk_threshold_medium,
            "high": state.settings.risk_threshold_high,
        },
        validation_metrics=metadata.validation_metrics,
        holdout_metrics=metadata.holdout_metrics,
        feature_config=metadata.feature_config,
        hyperparameters=metadata.hyperparameters,
        library_versions=metadata.library_versions,
    )
