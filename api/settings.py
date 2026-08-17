"""API configuration from environment variables.

No secrets or hostnames are hardcoded: everything comes from the environment,
with defaults that work for a local run and are overridden by
``docker-compose.yml`` inside the container network.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the prediction service."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    app_env: str = "local"
    log_level: str = "INFO"

    # Model loading
    model_artifact_path: str = "models/model_artifact.pkl"
    model_registry_name: str = "fraud-detector"
    model_version: str = "1"
    mlflow_tracking_uri: str | None = None
    load_from_registry: bool = False

    # Risk banding — thresholds are configuration, not constants, because the
    # operating point is a business decision that changes without retraining.
    risk_threshold_medium: float = Field(default=0.30, ge=0.0, le=1.0)
    risk_threshold_high: float = Field(default=0.70, ge=0.0, le=1.0)

    # PostgreSQL
    postgres_user: str = "fraud"
    postgres_password: str = "fraud"
    postgres_db: str = "fraud_platform"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    enable_prediction_log: bool = True

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    prediction_cache_ttl_seconds: int = 300
    rate_limit_requests_per_minute: int = 120
    enable_redis: bool = True
    velocity_history_seconds: int = 604_800  # widest training window (168 h)

    # Batch limits — an unbounded batch endpoint is a denial-of-service vector.
    max_batch_size: int = 500

    @property
    def database_url(self) -> str:
        """SQLAlchemy connection URL."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — parsed once per process."""
    return Settings()
