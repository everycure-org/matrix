import logging
import os
from typing import Union

import mlflow

logger = logging.getLogger(__name__)


def log_metric(context, name, value: Union[int, float]):
    disable_mlflow = (
        os.getenv("KEDRO_HOOKS_DISABLE_MLFLOW", "false").lower() == "true"
        or os.getenv("KEDRO_HOOKS_DISABLE_MLFLOW_KEDRO", "false").lower() == "true"
        or os.getenv("KEDRO_HOOKS_DISABLE_RELEASE", "false").lower() == "true"
    )

    logger.info(f"{name}: {value}")
    # TODO: can we figure out context so we don't have to pass it.
    # What kedro node am I running? Can I set this as my "context" variable
    mlflow.log_metric(f"{context}/{name}", value)

    if not disable_mlflow:
        mlflow.log_metric(f"{context}/{name}", value)
    else:
        logger.info("MLFlow logging is disabled via environment variables.")
