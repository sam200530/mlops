"""SQLAlchemy models for prediction logging.

**What is deliberately not stored.** A fraud service's own logs are a payment-data
liability, so raw card and identity values never land in the database. ``card1``
and the derived entity key are stored as salted SHA-256 digests: that still
supports "how many predictions did this card generate" and per-entity drift
analysis, without the table becoming a card database. Email domains, device
strings, and the ``id_*`` identity block are not persisted at all.

**What is stored, and why each field earns its place.** Everything here feeds
either monitoring or incident investigation: the probability and risk level for
score-distribution drift, latency for performance monitoring, ``features_supplied``
for input-completeness monitoring, ``cache_hit`` to verify Redis is doing useful
work, and a small numeric ``feature_summary`` for feature-drift tests.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all tables."""


def hash_identifier(value: object, salt: str) -> str | None:
    """Salted digest of an identifier, or ``None`` if absent.

    The salt comes from configuration and is not committed, so digests are not
    reversible via a precomputed table of the ~18k possible ``card1`` values —
    without a salt, hashing such a small domain would be trivially invertible and
    would provide no real protection.
    """
    if value is None or value == "":
        return None
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:32]


class PredictionLog(Base):
    """One scored transaction."""

    __tablename__ = "prediction_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)

    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    endpoint: Mapped[str] = mapped_column(String(32), nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Non-sensitive request characteristics used by monitoring.
    transaction_amt: Mapped[float | None] = mapped_column(Float, nullable=True)
    product_cd: Mapped[str | None] = mapped_column(String(8), nullable=True)
    features_supplied: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Pseudonymous entity references.
    card_hash: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    entity_hash: Mapped[str | None] = mapped_column(String(32), nullable=True)

    #: Small numeric summary of engineered features, for drift tests.
    feature_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (Index("ix_prediction_log_created_model", "created_at", "model_version"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<PredictionLog {self.request_id} p={self.fraud_probability:.4f} "
            f"{self.risk_level} {self.latency_ms:.1f}ms>"
        )


class ValidationFailureLog(Base):
    """A request rejected before scoring.

    Tracked separately because input-validation failure rate is a monitoring
    signal in its own right: a sudden spike usually means an upstream caller
    changed its payload, which would otherwise show up much later as unexplained
    feature drift.
    """

    __tablename__ = "validation_failure_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    endpoint: Mapped[str] = mapped_column(String(32), nullable=False)
    error_type: Mapped[str] = mapped_column(String(64), nullable=False)
    detail: Mapped[str] = mapped_column(String(1024), nullable=False)


def utc_now() -> datetime:
    """Timezone-aware current time."""
    return datetime.now(UTC)
