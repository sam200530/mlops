"""Online velocity features backed by Redis.

This is the component that makes the serving path *honest*. The model's strongest
learned signals include trailing-window counts per card — "how many times has
this card transacted in the last hour". A stateless API cannot compute that from
a single request body, and the usual shortcut is to send NaN and quietly serve a
weaker model than the one that was evaluated.

Instead the service keeps a short rolling history of observed transactions per
entity in Redis sorted sets, scored by timestamp. Each request queries its own
trailing windows, then records itself for the benefit of later requests. This
reproduces the training-time definition exactly: **only strictly earlier
transactions count**, because the current one is recorded after the query.

Redis is the right store for this and not a decorative dependency:

* sorted sets give O(log n + m) range queries by timestamp, which is precisely
  the window operation needed;
* ``ZREMRANGEBYSCORE`` expires history older than the widest training window
  (168 h) so memory stays bounded without a background job;
* state is shared across API workers — per-process memory would give different
  answers depending on which worker served the request.

The in-memory fallback exists so the service (and its tests) run without Redis.
It is per-process and therefore explicitly *not* production-correct, which
:meth:`VelocityStore.backend` reports so ``/health`` can tell the truth.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Entity key columns, matching src.features.builders.ENTITY_KEY_COLUMNS. The
#: feature names emitted here must match training exactly, so the short names
#: are derived the same way (leading underscore stripped).
ENTITY_COLUMNS = ("_entity_card", "_entity_card_addr", "_entity_card_full")


class RedisLike(Protocol):
    """Minimal Redis surface used here (keeps this module testable)."""

    def zadd(self, name: str, mapping: dict[str, float]) -> Any: ...
    def zrangebyscore(self, name: str, min: float, max: float) -> list[Any]: ...
    def zremrangebyscore(self, name: str, min: float, max: float) -> Any: ...
    def expire(self, name: str, seconds: int) -> Any: ...


def compute_entity_keys(record: dict[str, Any]) -> dict[str, int]:
    """Derive entity keys from a raw record.

    Must stay identical to :func:`src.features.builders.add_entity_keys`, so the
    online key for a card matches the offline one. Missing parts map to 0.
    """

    def as_int(value: Any) -> int:
        if value is None:
            return 0
        try:
            if value != value:  # NaN
                return 0
        except TypeError:
            return 0
        return int(value)

    card1 = as_int(record.get("card1"))
    addr1 = as_int(record.get("addr1"))
    card2 = as_int(record.get("card2"))
    return {
        "_entity_card": card1,
        "_entity_card_addr": card1 * 1_000 + addr1,
        "_entity_card_full": card1 * 1_000_000 + addr1 * 1_000 + card2,
    }


class VelocityStore:
    """Rolling per-entity transaction history for online velocity features."""

    def __init__(
        self,
        redis_client: RedisLike | None,
        windows_hours: tuple[int, ...] = (1, 24, 168),
        history_seconds: int = 604_800,
    ) -> None:
        self.redis = redis_client
        self.windows_hours = windows_hours
        self.history_seconds = history_seconds
        # Fallback: {redis_key: [(timestamp, amount), ...]}
        self._memory: dict[str, list[tuple[float, float]]] = defaultdict(list)

    @property
    def backend(self) -> str:
        """Which backend is in use — surfaced by ``/health``."""
        return "redis" if self.redis is not None else "in_memory_fallback"

    @staticmethod
    def _key(entity_column: str, entity_value: int) -> str:
        return f"vel:{entity_column.lstrip('_')}:{entity_value}"

    def features_for(self, record: dict[str, Any], timestamp: int) -> dict[str, float]:
        """Velocity features for one transaction, from strictly earlier history.

        Args:
            record: Raw transaction record (needs ``card1``/``addr1``/``card2``).
            timestamp: Transaction time in the dataset's seconds base.

        Returns:
            Mapping of velocity feature name to value, matching training names.
        """
        entity_keys = compute_entity_keys(record)
        amount = float(record.get("TransactionAmt") or 0.0)
        features: dict[str, float] = {}

        for column, value in entity_keys.items():
            key = self._key(column, value)
            short = column.lstrip("_")
            history = self._read(key, timestamp)

            previous_timestamps = [ts for ts, _ in history]
            features[f"{short}_seconds_since_prev"] = (
                float(timestamp - max(previous_timestamps)) if previous_timestamps else float("nan")
            )

            for hours in self.windows_hours:
                window_start = timestamp - hours * 3600
                in_window = [(ts, amt) for ts, amt in history if ts > window_start]
                count = float(len(in_window))
                total = float(sum(amt for _, amt in in_window))
                features[f"{short}_txn_count_{hours}h"] = count
                features[f"{short}_amt_sum_{hours}h"] = total
                features[f"{short}_amt_mean_{hours}h"] = (
                    total / count if count > 0 else float("nan")
                )

        self._record(entity_keys, timestamp, amount)
        return features

    def _read(self, key: str, timestamp: int) -> list[tuple[float, float]]:
        """History strictly earlier than ``timestamp`` within the retention window."""
        low = timestamp - self.history_seconds
        if self.redis is not None:
            try:
                # Exclusive upper bound: the current transaction must not count
                # itself, matching the offline definition.
                raw = self.redis.zrangebyscore(key, low, f"({timestamp}")
                return [self._parse(member) for member in raw]
            except Exception as error:  # noqa: BLE001 - degrade, never fail a prediction
                logger.warning("Redis velocity read failed for %s: %s", key, error)
                return []
        return [(ts, amt) for ts, amt in self._memory[key] if low <= ts < timestamp]

    def _record(self, entity_keys: dict[str, int], timestamp: int, amount: float) -> None:
        """Append this transaction to each entity's history and trim old entries."""
        nonce = f"{time.time_ns()}"
        for column, value in entity_keys.items():
            key = self._key(column, value)
            if self.redis is not None:
                try:
                    self.redis.zadd(key, {f"{timestamp}|{amount}|{nonce}": float(timestamp)})
                    self.redis.zremrangebyscore(key, 0, timestamp - self.history_seconds)
                    self.redis.expire(key, self.history_seconds)
                except Exception as error:  # noqa: BLE001
                    logger.warning("Redis velocity write failed for %s: %s", key, error)
            else:
                bucket = self._memory[key]
                bucket.append((float(timestamp), amount))
                cutoff = timestamp - self.history_seconds
                self._memory[key] = [(ts, amt) for ts, amt in bucket if ts >= cutoff]

    @staticmethod
    def _parse(member: Any) -> tuple[float, float]:
        """Parse a stored member back into ``(timestamp, amount)``."""
        text = member.decode() if isinstance(member, bytes) else str(member)
        parts = text.split("|")
        try:
            return float(parts[0]), float(parts[1])
        except (IndexError, ValueError):
            return 0.0, 0.0

    def reset(self) -> None:
        """Clear the in-memory fallback (used by tests)."""
        self._memory.clear()
