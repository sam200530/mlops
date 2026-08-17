"""Live monitoring endpoints.

These serve *observed service behaviour* — request counts, latency, score
distribution, validation failures. Offline drift analysis against the training
reference lives in ``scripts/monitor.py``, because it needs the full reference
distribution and is not something to compute inside an HTTP request.

The traffic these numbers describe is local or simulated. Nothing here is
production traffic, and the report says so explicitly.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from api.dependencies import AppState, get_state
from database.models import PredictionLog
from database.session import session_scope

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

#: Minimum scored requests before live score behaviour is reported at all.
#: Mirrors service.min_scores_for_drift in configs/monitoring.yaml.
MIN_SCORES_FOR_DRIFT = 30


@router.get("/metrics")
def metrics(state: AppState = Depends(get_state)) -> dict:
    """Service counters, latency percentiles and score distribution."""
    counters = state.metrics.counters()
    hits = counters.get("cache_hits", 0.0)
    misses = counters.get("cache_misses", 0.0)
    total_cache = hits + misses

    return {
        "traffic_source": "local_or_simulated",
        "uptime_seconds": round(state.metrics.uptime_seconds(), 1),
        "model_version": state.model_version,
        "counters": counters,
        "latency_ms": state.metrics.latency_summary(),
        "fraud_probability": state.metrics.score_summary(),
        "cache_hit_rate": round(hits / total_cache, 4) if total_cache else None,
        "metrics_backend": state.metrics.backend,
        "velocity_backend": (
            state.velocity_store.backend if state.velocity_store is not None else "unavailable"
        ),
    }


@router.get("/prediction-drift")
def score_drift(state: AppState = Depends(get_state)) -> dict:
    """Observed live score behaviour, compared against training-time reference values.

    Score behaviour is the only degradation signal available without labels:
    chargebacks arrive weeks after the transaction, so waiting for measured PR-AUC
    to move means noticing far too late, whereas the model's own output
    distribution moves immediately.

    **No PSI is computed here, deliberately.** PSI needs the training score
    *distribution* as its reference, and the service carries only summary metrics
    from training, not the raw validation scores. Comparing live scores against a
    single summary number would yield a figure that looks like drift detection
    without being it. ``scripts/monitor.py`` holds the real reference and computes
    feature- and score-level PSI properly.
    """
    reference = state.artifact.metadata.validation_metrics if state.artifact is not None else {}
    recent = state.metrics.recent_scores()
    if len(recent) < MIN_SCORES_FOR_DRIFT:
        return {
            "status": "insufficient_data",
            "n_recent_scores": len(recent),
            "message": (
                f"At least {MIN_SCORES_FOR_DRIFT} scored requests are needed. "
                "Run scripts/simulate_traffic.py."
            ),
        }

    flag_rate = None
    if state.artifact is not None:
        threshold = state.artifact.decision_threshold
        flag_rate = round(sum(1 for s in recent if s >= threshold) / len(recent), 6)

    return {
        "traffic_source": "local_or_simulated",
        "status": "ok",
        "n_recent_scores": len(recent),
        "recent_score_summary": state.metrics.score_summary(),
        "observed_flag_rate": flag_rate,
        "training_validation_prevalence": reference.get("prevalence"),
        "training_validation_pr_auc": reference.get("pr_auc"),
        "note": (
            "Score and feature PSI against the training reference distribution are "
            "computed by scripts/monitor.py. This endpoint reports observed live "
            "behaviour only."
        ),
    }


@router.get("/predictions/summary")
def prediction_summary(limit: int = 1000, state: AppState = Depends(get_state)) -> dict:
    """Aggregate the persisted prediction log.

    Reads from Postgres rather than Redis because this is a historical question
    over durable records, which is exactly what a relational store is for.
    """
    if not state.settings.enable_prediction_log:
        return {"status": "disabled", "message": "Prediction logging is disabled"}

    with session_scope() as session:
        if session is None:
            return {"status": "unavailable", "message": "Database not reachable"}
        total = session.scalar(select(func.count()).select_from(PredictionLog)) or 0
        by_risk = dict(
            session.execute(
                select(PredictionLog.risk_level, func.count()).group_by(PredictionLog.risk_level)
            ).all()
        )
        stats = session.execute(
            select(
                func.avg(PredictionLog.fraud_probability),
                func.max(PredictionLog.fraud_probability),
                func.avg(PredictionLog.latency_ms),
                func.count().filter(PredictionLog.cache_hit.is_(True)),
            )
        ).one()

        return {
            "traffic_source": "local_or_simulated",
            "total_predictions": int(total),
            "by_risk_level": {str(k): int(v) for k, v in by_risk.items()},
            "mean_fraud_probability": round(float(stats[0]), 6) if stats[0] is not None else None,
            "max_fraud_probability": round(float(stats[1]), 6) if stats[1] is not None else None,
            "mean_latency_ms": round(float(stats[2]), 2) if stats[2] is not None else None,
            "cache_hits": int(stats[3] or 0),
            "limit_applied": limit,
        }
