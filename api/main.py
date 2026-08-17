"""FastAPI application entry point.

The model, Redis connection and database engine are initialised once in the
lifespan handler and reused for every request. Startup deliberately does **not**
fail when the model is missing — the service comes up and reports ``degraded`` on
``/health`` with a 503 from the prediction endpoints, which is far easier to
diagnose in a container than a crash-loop with no HTTP surface.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.dependencies import connect_redis, load_artifact, state
from api.routers import explain, health, monitoring, predict
from api.settings import get_settings
from api.velocity_store import VelocityStore
from database.session import create_tables, dispose_engine, init_engine
from src.monitoring.metrics_store import MetricsStore
from src.utils.logging_config import setup_logging

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Load the model and connect dependencies once, at startup."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting fraud detection API (env=%s)", settings.app_env)

    state.settings = settings
    state.redis_client = connect_redis(settings)
    state.metrics = MetricsStore(state.redis_client)
    state.velocity_store = VelocityStore(
        state.redis_client, history_seconds=settings.velocity_history_seconds
    )

    try:
        state.artifact = load_artifact(settings)
        logger.info(
            "Model loaded: %s trained %s, %d features, threshold %.4f",
            state.artifact.metadata.model_name,
            state.artifact.metadata.trained_at,
            state.artifact.metadata.n_features,
            state.artifact.decision_threshold,
        )
    except FileNotFoundError:
        logger.error(
            "No model artifact found at %s — the API will report degraded health and "
            "return 503 from prediction endpoints. Run scripts/train.py.",
            settings.model_artifact_path,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Model load failed — starting in degraded mode")

    if settings.enable_prediction_log:
        try:
            init_engine(settings.database_url)
            create_tables()
            state.database_ready = True
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Database unavailable (%s) — prediction logging disabled for this process", error
            )
            state.database_ready = False

    yield

    logger.info("Shutting down")
    dispose_engine()
    if state.redis_client is not None:
        # A failure closing the connection during shutdown is not actionable.
        with contextlib.suppress(Exception):
            state.redis_client.close()


app = FastAPI(
    title="Fraud Detection & ML Decision Platform",
    description=(
        "Production-style fraud scoring service for the IEEE-CIS dataset. "
        "Serves a LightGBM model with SHAP explanations, Redis-backed velocity "
        "features, and prediction logging to PostgreSQL."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(monitoring.router)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Attach a request id and a duration header to every response."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a structured 422 and count the failure.

    Input-validation failure rate is a monitoring signal in its own right: a spike
    almost always means an upstream caller changed its payload, which would
    otherwise surface much later as unexplained feature drift.
    """
    state.metrics.increment("validation_failures")
    logger.warning("Validation failure on %s: %s", request.url.path, exc.errors()[:3])
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request validation failed",
            "error_type": "RequestValidationError",
            "errors": [
                {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                for e in exc.errors()[:10]
            ],
        },
    )


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Point callers at the docs."""
    return {
        "service": "fraud-detection-platform",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
    }
