"""MLflow pyfunc wrapper for the model bundle.

Purpose is registry semantics: a named model with real, incrementing versions and
stage transitions, so "which model is serving" is an auditable pointer rather than
whichever file is on the volume.

Scope note, stated plainly: this wrapper's ``predict`` expects a frame that has
already been through ``FeaturePipeline.prepare`` — i.e. causal features plus
velocity columns. It does **not** accept raw API payloads, because velocity
features require the per-entity history that lives in Redis, and a pyfunc has no
access to it. Serving raw payloads through this wrapper would silently produce a
weaker model than the one that was evaluated. The FastAPI service is therefore the
supported serving path; the registry is the versioning and provenance mechanism.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd

from src.models.artifact import ModelArtifact

logger = logging.getLogger(__name__)

ARTIFACT_KEY = "model_artifact"


class FraudDetectionModel(mlflow.pyfunc.PythonModel):
    """Serves calibrated fraud probabilities from a :class:`ModelArtifact`."""

    def load_context(self, context: Any) -> None:
        """Load the bundle once when MLflow instantiates the model."""
        path = Path(context.artifacts[ARTIFACT_KEY])
        self.artifact = ModelArtifact.load(path)
        logger.info("pyfunc loaded artifact %s", self.artifact.metadata.model_name)

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict | None = None):  # noqa: ARG002
        """Calibrated fraud probability per row of a *prepared* frame."""
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("model_input must be a pandas DataFrame of prepared features")
        return self.artifact.predict_proba(model_input)


def _pip_requirements() -> list[str]:
    """Pinned runtime requirements recorded with the logged model.

    Pinned to the versions actually installed, so a model logged today can be
    reproduced later rather than resolving to whatever is newest.
    """
    import lightgbm
    import numpy
    import pandas
    import shap
    import sklearn

    return [
        f"numpy=={numpy.__version__}",
        f"pandas=={pandas.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"lightgbm=={lightgbm.__version__}",
        f"shap=={shap.__version__}",
        f"mlflow=={mlflow.__version__}",
    ]


def log_and_register(
    artifact_path: Path,
    registry_name: str,
    metadata: dict[str, Any] | None = None,
    register: bool = True,
) -> tuple[str, str | None]:
    """Log the bundle as a pyfunc model and optionally register a new version.

    Returns:
        ``(model_uri, registered_version)``; version is ``None`` when the
        registry is unavailable (e.g. a file-based tracking store without a
        registry backend).
    """
    # Requirements are declared explicitly rather than inferred, so the recorded
    # environment is the one we chose and pinned rather than whatever MLflow
    # discovers by introspecting the process at log time.
    logged = mlflow.pyfunc.log_model(
        name="fraud_model",
        python_model=FraudDetectionModel(),
        artifacts={ARTIFACT_KEY: str(artifact_path)},
        metadata=metadata or {},
        pip_requirements=_pip_requirements(),
    )
    logger.info("Logged pyfunc model at %s", logged.model_uri)

    version: str | None = None
    if register:
        try:
            registered = mlflow.register_model(model_uri=logged.model_uri, name=registry_name)
            version = str(registered.version)
            logger.info("Registered %s version %s", registry_name, version)
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Model registry unavailable (%s) — model logged but not registered", error
            )

    return logged.model_uri, version
