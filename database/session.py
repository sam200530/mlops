"""Database engine and session management.

The API must remain able to score transactions when Postgres is unavailable:
prediction logging is valuable but it is not on the critical path of an
authorisation decision. So engine creation is lazy and failures degrade to
"logging disabled" rather than propagating to the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def init_engine(database_url: str, echo: bool = False) -> Engine:
    """Create the engine and session factory (idempotent)."""
    global _engine, _session_factory
    if _engine is None:
        _engine = create_engine(
            database_url,
            echo=echo,
            pool_pre_ping=True,  # survive a Postgres restart without manual intervention
            pool_size=5,
            max_overflow=5,
            future=True,
        )
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        logger.info("Database engine initialised")
    return _engine


def create_tables(engine: Engine | None = None) -> None:
    """Create tables if absent. Safe to call on every startup."""
    target = engine or _engine
    if target is None:
        raise RuntimeError("init_engine must be called before create_tables")
    Base.metadata.create_all(target)
    logger.info("Database tables ensured: %s", ", ".join(Base.metadata.tables))


def check_connection() -> bool:
    """Whether the database is reachable right now."""
    if _engine is None:
        return False
    try:
        with _engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as error:  # noqa: BLE001
        logger.warning("Database health check failed: %s", error)
        return False


@contextmanager
def session_scope() -> Iterator[Session | None]:
    """Transactional session scope that never breaks a prediction.

    Yields ``None`` when the database is unconfigured, and rolls back on error
    while logging it, so a logging failure cannot turn a successful prediction
    into a 500.
    """
    if _session_factory is None:
        yield None
        return

    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception as error:  # noqa: BLE001
        session.rollback()
        logger.warning("Database write failed, rolled back: %s", error)
    finally:
        session.close()


def dispose_engine() -> None:
    """Dispose the engine on shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        logger.info("Database engine disposed")
    _engine = None
    _session_factory = None
