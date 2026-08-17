"""Shared application state and request-scoped dependencies.

The model is loaded **once**, at application startup, into module-level state and
reused for every request. Loading per request would add hundreds of milliseconds
and re-read the artifact from disk on every call.

Also here: the raw-record -> prepared-frame construction that the prediction and
explanation endpoints share, plus the Redis-backed cache and rate limiter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException, Request, status

from api.settings import Settings, get_settings
from api.velocity_store import VelocityStore
from src.data.schema import KEY, RAW_CATEGORICAL, TIME_COL
from src.models.artifact import ARTIFACT_FILENAME, ModelArtifact
from src.monitoring.metrics_store import MetricsStore
from src.utils.paths import ROOT

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Process-wide state populated during startup."""

    settings: Settings
    artifact: ModelArtifact | None = None
    velocity_store: VelocityStore | None = None
    metrics: MetricsStore = field(default_factory=MetricsStore)
    redis_client: Any | None = None
    database_ready: bool = False
    started_at: float = field(default_factory=time.time)

    @property
    def model_version(self) -> str:
        return self.settings.model_version


state = AppState(settings=get_settings())


# --- lifecycle ------------------------------------------------------------


def load_artifact(settings: Settings) -> ModelArtifact:
    """Load the model bundle from the registry if configured, else from disk.

    Registry loading is preferred in a real deployment because the version is
    then an auditable pointer rather than whatever file happens to be on the
    volume. Local-file loading is the fallback so the service runs without an
    MLflow server.
    """
    if settings.load_from_registry and settings.mlflow_tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            uri = f"models:/{settings.model_registry_name}/{settings.model_version}"
            local_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
            # MLflow returns the pyfunc model directory; the bundle itself is
            # stored under the artifacts subdirectory the pyfunc wrapper declared.
            candidates = [
                local_dir,
                local_dir / "artifacts",
                *sorted(local_dir.rglob(ARTIFACT_FILENAME)),
            ]
            for candidate in candidates:
                target = candidate if candidate.is_file() else candidate / ARTIFACT_FILENAME
                if target.is_file():
                    logger.info("Loaded model from registry %s (%s)", uri, target)
                    return ModelArtifact.load(target)
            raise FileNotFoundError(f"{ARTIFACT_FILENAME} not found under {local_dir}")
        except Exception as error:  # noqa: BLE001
            logger.error("Registry load failed (%s) — falling back to local artifact", error)

    path = ROOT / settings.model_artifact_path
    return ModelArtifact.load(path)


def connect_redis(settings: Settings) -> Any | None:
    """Connect to Redis, returning ``None`` if unavailable.

    Redis is an enhancement (cache, rate limit, velocity, metrics), never a
    prerequisite for scoring, so an outage must degrade rather than fail.
    """
    if not settings.enable_redis:
        logger.info("Redis disabled by configuration")
        return None
    try:
        import redis

        client = redis.Redis.from_url(
            settings.redis_url, socket_connect_timeout=2, socket_timeout=2, decode_responses=False
        )
        client.ping()
        logger.info("Connected to Redis at %s", settings.redis_url)
        return client
    except Exception as error:  # noqa: BLE001
        logger.warning("Redis unavailable (%s) — using in-process fallbacks", error)
        return None


# --- dependencies ---------------------------------------------------------


def get_state() -> AppState:
    """Application state accessor."""
    return state


def require_artifact() -> ModelArtifact:
    """Loaded model, or 503 if startup did not complete."""
    if state.artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train a model with scripts/train.py first.",
        )
    return state.artifact


def enforce_rate_limit(request: Request) -> None:
    """Fixed-window per-client rate limit.

    Implemented in Redis rather than process memory because the limit must hold
    across API workers — four workers each allowing 120 requests/min would in
    fact allow 480. Fails open: if Redis is down, requests are served rather than
    rejected, since dropping legitimate authorisations is worse than briefly
    losing rate limiting.
    """
    settings = state.settings
    if state.redis_client is None or settings.rate_limit_requests_per_minute <= 0:
        return

    client = request.headers.get("x-api-key") or (
        request.client.host if request.client else "unknown"
    )
    window = int(time.time() // 60)
    key = f"ratelimit:{client}:{window}"
    try:
        pipe = state.redis_client.pipeline()
        pipe.incr(key, 1)
        pipe.expire(key, 120)
        count = int(pipe.execute()[0])
    except Exception as error:  # noqa: BLE001
        logger.warning("Rate limit check failed, allowing request: %s", error)
        return

    if count > settings.rate_limit_requests_per_minute:
        state.metrics.increment("rate_limited_requests")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {settings.rate_limit_requests_per_minute} requests/minute"
            ),
            headers={"Retry-After": "60"},
        )


# --- prediction cache -----------------------------------------------------


def cache_key(record: dict[str, Any]) -> str:
    """Stable digest of a raw record, used as the cache key."""
    payload = json.dumps(record, sort_keys=True, default=str)
    return "predcache:" + hashlib.sha256(payload.encode()).hexdigest()[:32]


def cache_get(key: str) -> dict[str, Any] | None:
    """Look up a cached prediction."""
    if state.redis_client is None:
        return None
    try:
        raw = state.redis_client.get(key)
        return json.loads(raw) if raw else None
    except Exception as error:  # noqa: BLE001
        logger.warning("Cache read failed: %s", error)
        return None


def cache_set(key: str, value: dict[str, Any]) -> None:
    """Store a prediction with the configured TTL."""
    if state.redis_client is None:
        return
    try:
        state.redis_client.setex(
            key, state.settings.prediction_cache_ttl_seconds, json.dumps(value)
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("Cache write failed: %s", error)


# --- feature frame construction -------------------------------------------


def build_prepared_frame(
    records: list[dict[str, Any]],
    artifact: ModelArtifact,
    velocity_store: VelocityStore,
    default_timestamp: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Turn raw request records into a frame the feature pipeline accepts.

    Steps:
      1. Materialise every raw column the pipeline reads, filling omissions with
         NaN (which the model handles natively).
      2. Assign a synthetic ``TransactionID`` — needed as the velocity join key
         and nothing else; it is never a model input.
      3. Sort chronologically, since velocity accumulates in time order.
      4. Ask the velocity store for trailing-window features per row, which also
         records each transaction for subsequent requests.
      5. Run the causal feature builders.

    Returns:
        ``(prepared, original_positions)`` where ``original_positions[i]`` is the
        index in ``records`` of prepared row ``i``. Callers must use it to map
        predictions back to the submitted order.
    """
    columns = artifact.metadata.raw_input_columns or []
    if not columns:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model artifact does not declare raw_input_columns; retrain with scripts/train.py",
        )

    categorical = set(RAW_CATEGORICAL)
    frame: dict[str, Any] = {}
    n_rows = len(records)

    for column in columns:
        if column in categorical:
            frame[column] = pd.Series(
                [_string_or_none(record.get(column)) for record in records], dtype="object"
            )
        elif column == KEY:
            continue
        elif column == TIME_COL:
            frame[column] = pd.Series(
                [int(record.get(TIME_COL) or default_timestamp) for record in records],
                dtype="int64",
            )
        else:
            frame[column] = pd.Series(
                [_float_or_nan(record.get(column)) for record in records], dtype="float32"
            )

    df = pd.DataFrame(frame)
    # Synthetic, monotonic ids: unique within the request and stable in ordering.
    base_id = int(time.time_ns() // 1000)
    df.insert(0, KEY, np.arange(base_id, base_id + n_rows, dtype="int64"))

    # Velocity must be accumulated in chronological order, so the frame is sorted
    # by timestamp — which means prepared-row order is NOT request order when a
    # batch arrives out of order. The caller needs the mapping back, or batch
    # responses would be silently paired with the wrong transactions.
    df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    original_positions = (df[KEY].to_numpy() - base_id).astype("int64")

    velocity_rows: list[dict[str, float]] = []
    for position in range(len(df)):
        record = {
            "card1": df.at[position, "card1"] if "card1" in df.columns else None,
            "addr1": df.at[position, "addr1"] if "addr1" in df.columns else None,
            "card2": df.at[position, "card2"] if "card2" in df.columns else None,
            "TransactionAmt": df.at[position, "TransactionAmt"],
        }
        features = velocity_store.features_for(record, int(df.at[position, TIME_COL]))
        features[KEY] = int(df.at[position, KEY])
        velocity_rows.append(features)

    velocity_frame = pd.DataFrame(velocity_rows).set_index(KEY)
    prepared = artifact.feature_pipeline.prepare(df, velocity_frame=velocity_frame)
    return prepared, original_positions


def count_supplied_features(record: dict[str, Any]) -> int:
    """How many non-null raw features the caller actually provided."""
    return sum(1 for value in record.values() if value is not None and value == value)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        pass
    return str(value)


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
