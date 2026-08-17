"""Prediction endpoints."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from api.dependencies import (
    AppState,
    build_prepared_frame,
    cache_get,
    cache_key,
    cache_set,
    count_supplied_features,
    enforce_rate_limit,
    get_state,
    require_artifact,
)
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    TransactionRequest,
)
from database.models import PredictionLog, hash_identifier
from database.session import session_scope
from src.models.artifact import ModelArtifact

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"], dependencies=[Depends(enforce_rate_limit)])


def _log_prediction(
    request_id: str,
    record: dict[str, Any],
    probability: float,
    risk: str,
    latency_ms: float,
    endpoint: str,
    artifact: ModelArtifact,
    state: AppState,
    cache_hit: bool,
    feature_summary: dict[str, float] | None = None,
) -> None:
    """Persist one prediction. Runs in a background task, off the response path."""
    if not state.settings.enable_prediction_log:
        return
    salt = state.settings.postgres_password  # not committed; see .env.example
    with session_scope() as session:
        if session is None:
            return
        session.add(
            PredictionLog(
                request_id=request_id,
                model_name=artifact.metadata.model_name,
                model_version=state.model_version,
                fraud_probability=float(probability),
                risk_level=risk,
                flagged=bool(probability >= artifact.decision_threshold),
                decision_threshold=float(artifact.decision_threshold),
                latency_ms=float(latency_ms),
                endpoint=endpoint,
                cache_hit=cache_hit,
                transaction_amt=_safe_float(record.get("TransactionAmt")),
                product_cd=_safe_str(record.get("ProductCD")),
                features_supplied=count_supplied_features(record),
                card_hash=hash_identifier(record.get("card1"), salt),
                entity_hash=hash_identifier(
                    f"{record.get('card1')}_{record.get('addr1')}_{record.get('card2')}", salt
                ),
                feature_summary=feature_summary,
            )
        )


def _feature_summary(prepared, position: int = 0) -> dict[str, float]:
    """Small numeric snapshot of engineered features, for drift monitoring."""
    interesting = [
        "n_missing_total",
        "identity_present",
        "log_amount",
        "hour_of_day",
        "entity_card_txn_count_1h",
        "entity_card_txn_count_24h",
        "entity_card_seconds_since_prev",
    ]
    summary: dict[str, float] = {}
    for name in interesting:
        if name in prepared.columns:
            value = prepared.iloc[position][name]
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric == numeric:  # skip NaN
                summary[name] = round(numeric, 4)
    return summary


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: TransactionRequest,
    background_tasks: BackgroundTasks,
    artifact: ModelArtifact = Depends(require_artifact),
    state: AppState = Depends(get_state),
) -> PredictionResponse:
    """Score a single transaction.

    A cache hit short-circuits both scoring *and* the velocity write. That is
    deliberate: a cache hit means this exact transaction was already scored
    within the TTL, i.e. a retry — so counting it again in the card's velocity
    history would inflate the very features the model relies on.
    """
    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    record = request.to_raw_record()
    state.metrics.increment("predict_requests")

    key = cache_key(record)
    cached = cache_get(key)
    if cached is not None:
        latency_ms = (time.perf_counter() - started) * 1000
        state.metrics.increment("cache_hits")
        state.metrics.observe_latency(latency_ms)
        state.metrics.observe_score(cached["fraud_probability"])
        background_tasks.add_task(
            _log_prediction,
            request_id,
            record,
            cached["fraud_probability"],
            cached["risk_level"],
            latency_ms,
            "predict",
            artifact,
            state,
            True,
            None,
        )
        return PredictionResponse(
            fraud_probability=cached["fraud_probability"],
            risk_level=cached["risk_level"],
            model_version=state.model_version,
            request_id=request_id,
            decision_threshold=artifact.decision_threshold,
            flagged=cached["fraud_probability"] >= artifact.decision_threshold,
            latency_ms=round(latency_ms, 2),
            features_supplied=count_supplied_features(record),
        )

    state.metrics.increment("cache_misses")
    try:
        prepared, _ = build_prepared_frame(
            [record],
            artifact,
            state.velocity_store,  # type: ignore[arg-type]
            default_timestamp=artifact.metadata.holdout_cut_dt,
        )
        probability = float(artifact.predict_proba(prepared)[0])
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        state.metrics.increment("scoring_errors")
        logger.exception("Scoring failed for request %s", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {type(error).__name__}",
        ) from error

    risk = artifact.risk_level(
        probability, state.settings.risk_threshold_medium, state.settings.risk_threshold_high
    )
    latency_ms = (time.perf_counter() - started) * 1000
    state.metrics.observe_latency(latency_ms)
    state.metrics.observe_score(probability)
    state.metrics.increment(f"risk_{risk}")

    cache_set(key, {"fraud_probability": probability, "risk_level": risk})
    background_tasks.add_task(
        _log_prediction,
        request_id,
        record,
        probability,
        risk,
        latency_ms,
        "predict",
        artifact,
        state,
        False,
        _feature_summary(prepared),
    )

    return PredictionResponse(
        fraud_probability=round(probability, 6),
        risk_level=risk,  # type: ignore[arg-type]
        model_version=state.model_version,
        request_id=request_id,
        decision_threshold=artifact.decision_threshold,
        flagged=probability >= artifact.decision_threshold,
        latency_ms=round(latency_ms, 2),
        features_supplied=count_supplied_features(record),
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    artifact: ModelArtifact = Depends(require_artifact),
    state: AppState = Depends(get_state),
) -> BatchPredictionResponse:
    """Score a batch in one vectorised pass.

    Batching matters because the per-request overhead here is dominated by frame
    construction and the feature transform, not the model — one pass over 500
    rows is far cheaper than 500 single calls. The cache is intentionally not
    consulted per row: partial cache hits would fragment the batch into separate
    transforms and lose exactly that advantage.
    """
    started = time.perf_counter()
    records = [transaction.to_raw_record() for transaction in request.transactions]
    state.metrics.increment("batch_requests")
    state.metrics.increment("predict_requests", len(records))

    if len(records) > state.settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size {len(records)} exceeds limit {state.settings.max_batch_size}",
        )

    try:
        prepared, original_positions = build_prepared_frame(
            records,
            artifact,
            state.velocity_store,  # type: ignore[arg-type]
            default_timestamp=artifact.metadata.holdout_cut_dt,
        )
        probabilities = artifact.predict_proba(prepared)
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        state.metrics.increment("scoring_errors")
        logger.exception("Batch scoring failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch scoring failed: {type(error).__name__}",
        ) from error

    total_latency_ms = (time.perf_counter() - started) * 1000
    per_row_latency = total_latency_ms / max(len(records), 1)

    # `prepared` is in chronological order, which is not the submitted order.
    # Responses are rebuilt in request order so predictions[i] always describes
    # transactions[i] — misaligning these would be a silent, severe defect.
    responses: list[PredictionResponse | None] = [None] * len(records)
    for prepared_position, probability in enumerate(probabilities):
        probability = float(probability)
        request_position = int(original_positions[prepared_position])
        risk = artifact.risk_level(
            probability, state.settings.risk_threshold_medium, state.settings.risk_threshold_high
        )
        request_id = str(uuid.uuid4())
        state.metrics.observe_score(probability)
        state.metrics.increment(f"risk_{risk}")
        record = records[request_position]
        background_tasks.add_task(
            _log_prediction,
            request_id,
            record,
            probability,
            risk,
            per_row_latency,
            "predict_batch",
            artifact,
            state,
            False,
            _feature_summary(prepared, prepared_position),
        )
        responses[request_position] = PredictionResponse(
            fraud_probability=round(probability, 6),
            risk_level=risk,  # type: ignore[arg-type]
            model_version=state.model_version,
            request_id=request_id,
            decision_threshold=artifact.decision_threshold,
            flagged=probability >= artifact.decision_threshold,
            latency_ms=round(per_row_latency, 2),
            features_supplied=count_supplied_features(record),
        )

    if any(response is None for response in responses):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal ordering error: not every transaction received a prediction",
        )

    state.metrics.observe_latency(total_latency_ms)
    return BatchPredictionResponse(
        predictions=[r for r in responses if r is not None],
        count=len(responses),
        model_version=state.model_version,
        latency_ms=round(total_latency_ms, 2),
    )


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _safe_str(value: Any) -> str | None:
    return None if value is None else str(value)[:8]
