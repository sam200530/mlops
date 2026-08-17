"""Register the trained model bundle in the MLflow Model Registry.

Run after ``scripts/train.py``. Creates a new registry version each time, so the
version served by the API is an auditable pointer rather than a file on a volume.

Usage:
    python scripts/register_model.py
    python scripts/register_model.py --stage Production --name fraud-detector
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.artifact import ARTIFACT_FILENAME, ModelArtifact  # noqa: E402
from src.models.mlflow_model import log_and_register  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.paths import MODELS_DIR, REPORTS_DIR  # noqa: E402

logger = logging.getLogger("register_model")


def main() -> int:
    parser = argparse.ArgumentParser(description="Register the model in MLflow.")
    parser.add_argument("--name", default=os.getenv("MODEL_REGISTRY_NAME", "fraud-detector"))
    parser.add_argument("--stage", default=None, help="Optional alias to assign (e.g. Production).")
    parser.add_argument("--no-register", action="store_true", help="Log only, do not register.")
    args = parser.parse_args()

    setup_logging()
    import mlflow

    artifact_path = MODELS_DIR / ARTIFACT_FILENAME
    if not artifact_path.is_file():
        logger.error("No artifact at %s — run scripts/train.py first", artifact_path)
        return 1

    artifact = ModelArtifact.load(artifact_path)
    uri = os.getenv("MLFLOW_TRACKING_URI") or (Path.cwd() / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("fraud-detection")
    logger.info("Tracking URI: %s", uri)

    metadata = artifact.metadata.to_dict()
    with mlflow.start_run(run_name=f"register_{artifact.metadata.model_name}"):
        mlflow.log_params(
            {
                "model_name": metadata["model_name"],
                "n_features": metadata["n_features"],
                "n_train_rows": metadata["n_train_rows"],
                "calibrated": metadata["calibrated"],
                "decision_threshold": artifact.decision_threshold,
            }
        )
        for key, value in (metadata.get("holdout_metrics") or {}).items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"holdout_{key}", value)
        for key, value in (metadata.get("validation_metrics") or {}).items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"validation_{key}", value)

        model_uri, version = log_and_register(
            artifact_path=artifact_path,
            registry_name=args.name,
            metadata={"decision_threshold": artifact.decision_threshold},
            register=not args.no_register,
        )

    result = {
        "registry_name": args.name,
        "model_uri": model_uri,
        "version": version,
        "tracking_uri": uri,
    }

    if version and args.stage:
        try:
            client = mlflow.MlflowClient()
            # Aliases replace the deprecated stage transitions in MLflow 2.9+.
            client.set_registered_model_alias(args.name, args.stage.lower(), version)
            result["alias"] = args.stage.lower()
            logger.info("Assigned alias %s -> version %s", args.stage.lower(), version)
        except Exception as error:  # noqa: BLE001
            logger.warning("Could not assign alias: %s", error)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "registered_model.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    logger.info("Registration result: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
