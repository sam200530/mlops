"""Live service counters, backed by Redis.

These are the operational metrics ``GET /monitoring/metrics`` serves: request
counts, latency percentiles, prediction distribution, validation failures, and
cache hit rate.

Redis rather than Postgres for this specific job because a metrics scrape should
not run an aggregate query over the whole prediction log — that cost grows with
traffic, and a monitoring endpoint that gets slower as the system gets busier is
worse than useless. Redis counters are O(1) writes and the latency reservoir is a
capped list, so scrape cost is constant.

Redis rather than process memory because the API runs multiple workers: per-process
counters would report one worker's slice of traffic as if it were the whole.

The in-memory fallback keeps the endpoint working without Redis (and in tests),
and reports itself as such so the numbers are never mistaken for cluster-wide.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Cap on retained latency/score samples per key. Bounded memory, and enough for
#: stable percentiles.
RESERVOIR_SIZE = 2000

KEY_PREFIX = "metrics"


class MetricsStore:
    """Counters, latency reservoir and score reservoir."""

    def __init__(self, redis_client: Any | None = None) -> None:
        self.redis = redis_client
        self._counters: dict[str, float] = defaultdict(float)
        self._latencies: deque[float] = deque(maxlen=RESERVOIR_SIZE)
        self._scores: deque[float] = deque(maxlen=RESERVOIR_SIZE)
        self._started_at = time.time()

    @property
    def backend(self) -> str:
        return "redis" if self.redis is not None else "in_memory_fallback"

    # --- writes -----------------------------------------------------------

    def increment(self, name: str, amount: float = 1.0) -> None:
        """Increment a named counter."""
        if self.redis is not None:
            try:
                self.redis.hincrbyfloat(f"{KEY_PREFIX}:counters", name, amount)
                return
            except Exception as error:  # noqa: BLE001 - metrics must never break a request
                logger.warning("Redis counter increment failed (%s): %s", name, error)
        self._counters[name] += amount

    def observe_latency(self, milliseconds: float) -> None:
        """Record a request latency sample."""
        self._push(f"{KEY_PREFIX}:latency", self._latencies, milliseconds)

    def observe_score(self, probability: float) -> None:
        """Record a predicted probability sample."""
        self._push(f"{KEY_PREFIX}:scores", self._scores, probability)

    def _push(self, key: str, fallback: deque[float], value: float) -> None:
        if self.redis is not None:
            try:
                pipe = self.redis.pipeline()
                pipe.lpush(key, float(value))
                pipe.ltrim(key, 0, RESERVOIR_SIZE - 1)
                pipe.execute()
                return
            except Exception as error:  # noqa: BLE001
                logger.warning("Redis reservoir push failed (%s): %s", key, error)
        fallback.append(float(value))

    # --- reads ------------------------------------------------------------

    def counters(self) -> dict[str, float]:
        """All counters."""
        if self.redis is not None:
            try:
                raw = self.redis.hgetall(f"{KEY_PREFIX}:counters") or {}
                return {_decode(k): float(v) for k, v in raw.items()}
            except Exception as error:  # noqa: BLE001
                logger.warning("Redis counter read failed: %s", error)
        return dict(self._counters)

    def latency_summary(self) -> dict[str, float]:
        """Latency percentiles in milliseconds."""
        return _percentiles(self._read(f"{KEY_PREFIX}:latency", self._latencies))

    def score_summary(self) -> dict[str, float]:
        """Predicted-probability distribution summary."""
        return _percentiles(self._read(f"{KEY_PREFIX}:scores", self._scores))

    def recent_scores(self) -> list[float]:
        """Raw retained scores, for drift comparison against the reference."""
        return list(self._read(f"{KEY_PREFIX}:scores", self._scores))

    def _read(self, key: str, fallback: deque[float]) -> list[float]:
        if self.redis is not None:
            try:
                return [float(_decode(v)) for v in (self.redis.lrange(key, 0, -1) or [])]
            except Exception as error:  # noqa: BLE001
                logger.warning("Redis reservoir read failed (%s): %s", key, error)
        return list(fallback)

    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    def reset(self) -> None:
        """Clear all metrics (used by tests)."""
        self._counters.clear()
        self._latencies.clear()
        self._scores.clear()
        if self.redis is not None:
            try:
                self.redis.delete(
                    f"{KEY_PREFIX}:counters", f"{KEY_PREFIX}:latency", f"{KEY_PREFIX}:scores"
                )
            except Exception as error:  # noqa: BLE001
                logger.warning("Redis metrics reset failed: %s", error)


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _percentiles(values: list[float]) -> dict[str, float]:
    """Summary statistics, or NaNs when there is no data yet."""
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    array = np.asarray(values, dtype="float64")
    return {
        "count": int(array.size),
        "mean": round(float(array.mean()), 4),
        "p50": round(float(np.percentile(array, 50)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "p99": round(float(np.percentile(array, 99)), 4),
        "max": round(float(array.max()), 4),
    }
