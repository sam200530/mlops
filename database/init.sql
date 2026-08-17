-- Schema initialisation for the prediction log.
--
-- Runs once, when Postgres initialises an empty data directory. The application
-- also calls SQLAlchemy's create_all() at startup, so the schema exists whether
-- the service runs under Docker Compose or directly against an existing
-- database; both paths are idempotent.
--
-- Privacy note: no raw card numbers, email domains, device strings or identity
-- fields are stored. card_hash and entity_hash are salted SHA-256 digests, so
-- per-entity analysis remains possible without the table becoming a card
-- database.

CREATE TABLE IF NOT EXISTS prediction_log (
    id                  BIGSERIAL PRIMARY KEY,
    request_id          VARCHAR(64)  NOT NULL,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),

    model_name          VARCHAR(64)  NOT NULL,
    model_version       VARCHAR(32)  NOT NULL,

    fraud_probability   DOUBLE PRECISION NOT NULL,
    risk_level          VARCHAR(16)  NOT NULL,
    flagged             BOOLEAN      NOT NULL DEFAULT FALSE,
    decision_threshold  DOUBLE PRECISION NOT NULL,

    latency_ms          DOUBLE PRECISION NOT NULL,
    endpoint            VARCHAR(32)  NOT NULL,
    cache_hit           BOOLEAN      NOT NULL DEFAULT FALSE,

    transaction_amt     DOUBLE PRECISION,
    product_cd          VARCHAR(8),
    features_supplied   INTEGER      NOT NULL DEFAULT 0,

    card_hash           VARCHAR(32),
    entity_hash         VARCHAR(32),

    feature_summary     JSONB,

    CONSTRAINT chk_probability_range CHECK (fraud_probability >= 0 AND fraud_probability <= 1),
    CONSTRAINT chk_risk_level CHECK (risk_level IN ('low', 'medium', 'high'))
);

-- Monitoring queries filter by time and group by model version, so this
-- composite index covers the common access pattern.
CREATE INDEX IF NOT EXISTS ix_prediction_log_created_model
    ON prediction_log (created_at DESC, model_version);

CREATE INDEX IF NOT EXISTS ix_prediction_log_request_id ON prediction_log (request_id);
CREATE INDEX IF NOT EXISTS ix_prediction_log_risk_level ON prediction_log (risk_level);
CREATE INDEX IF NOT EXISTS ix_prediction_log_card_hash  ON prediction_log (card_hash);

-- Validation failures are tracked separately: their rate is a monitoring signal
-- on its own, usually indicating an upstream caller changed its payload.
CREATE TABLE IF NOT EXISTS validation_failure_log (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    request_id  VARCHAR(64),
    endpoint    VARCHAR(32)  NOT NULL,
    error_type  VARCHAR(64)  NOT NULL,
    detail      VARCHAR(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_validation_failure_created ON validation_failure_log (created_at DESC);
