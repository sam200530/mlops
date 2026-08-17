"""Explanation endpoint."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import (
    AppState,
    build_prepared_frame,
    enforce_rate_limit,
    get_state,
    require_artifact,
)
from api.schemas import (
    ExplanationResponse,
    FeatureContributionResponse,
    TransactionRequest,
)
from src.explainability.shap_explainer import ShapExplainer
from src.models.artifact import ModelArtifact

logger = logging.getLogger(__name__)

router = APIRouter(tags=["explainability"], dependencies=[Depends(enforce_rate_limit)])

#: Built once on first use and reused. Constructing a TreeExplainer walks the
#: whole ensemble, which is far too slow to repeat per request.
_explainer: ShapExplainer | None = None


def get_explainer(artifact: ModelArtifact) -> ShapExplainer:
    """Lazily construct and cache the SHAP explainer."""
    global _explainer
    if _explainer is None:
        _explainer = ShapExplainer(artifact.model, artifact.feature_pipeline.feature_names)
        logger.info("Initialised SHAP explainer")
    return _explainer


def reset_explainer() -> None:
    """Drop the cached explainer (used when the model is reloaded, and by tests)."""
    global _explainer
    _explainer = None


@router.post("/explain", response_model=ExplanationResponse)
def explain(
    request: TransactionRequest,
    top_n: int = 10,
    artifact: ModelArtifact = Depends(require_artifact),
    state: AppState = Depends(get_state),
) -> ExplanationResponse:
    """Score a transaction and return the factors that drove the score.

    An analyst cannot action "0.87". They can action "0.87, driven by nine
    transactions on this card in the last hour and a device never seen before".
    TreeSHAP is exact for tree ensembles and needs no background sample, which is
    what makes this viable inside a request rather than as a batch job.

    Note the probability here is the calibrated score, while ``base_value`` and
    the SHAP values are in the model's log-odds space — they explain the ranking,
    not the calibrated number, so they sum to the raw margin rather than to the
    probability.
    """
    if not 1 <= top_n <= 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="top_n must be between 1 and 50",
        )
    if artifact.linear_preprocessor is not None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "SHAP explanations are implemented for the tree model only; the loaded "
                "artifact is a linear model."
            ),
        )

    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    record = request.to_raw_record()
    state.metrics.increment("explain_requests")

    try:
        prepared, _ = build_prepared_frame(
            [record],
            artifact,
            state.velocity_store,  # type: ignore[arg-type]
            default_timestamp=artifact.metadata.holdout_cut_dt,
        )
        features = artifact.transform_features(prepared)
        probability = float(artifact.predict_proba(prepared)[0])
        explainer = get_explainer(artifact)
        contributions = explainer.explain_row(features, row=0, top_n=top_n)
        base_value = explainer.base_value
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        state.metrics.increment("explain_errors")
        logger.exception("Explanation failed for request %s", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {type(error).__name__}",
        ) from error

    latency_ms = (time.perf_counter() - started) * 1000
    state.metrics.observe_latency(latency_ms)

    risk = artifact.risk_level(
        probability, state.settings.risk_threshold_medium, state.settings.risk_threshold_high
    )
    return ExplanationResponse(
        fraud_probability=round(probability, 6),
        risk_level=risk,  # type: ignore[arg-type]
        model_version=state.model_version,
        request_id=request_id,
        base_value=round(base_value, 6),
        top_factors=[
            FeatureContributionResponse(**contribution.to_dict()) for contribution in contributions
        ],
        latency_ms=round(latency_ms, 2),
    )
