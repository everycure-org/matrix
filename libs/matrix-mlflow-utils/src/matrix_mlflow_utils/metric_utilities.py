import logging
import os

import mlflow

logger = logging.getLogger(__name__)

DISABlE_MLFLOW = (
    os.getenv("KEDRO_HOOKS_DISABLE_MLFLOW", "false").lower() == "true"
    or os.getenv("KEDRO_HOOKS_DISABLE_MLFLOW_KEDRO", "false").lower() == "true"
    or os.getenv("KEDRO_HOOKS_DISABLE_RELEASE", "false").lower() == "true"
)


def log_metric(context: str, name: str, value: int | float):
    if not DISABlE_MLFLOW:
        mlflow.log_metric(f"{context}/{name}", value)
    else:
        logger.debug(f"{context}/{name}: {value}")
