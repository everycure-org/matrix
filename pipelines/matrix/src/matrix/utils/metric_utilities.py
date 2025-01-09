import logging
from typing import Union

import mlflow

logger = logging.getLogger(__name__)


def log_metric(context, name, value: Union[int, float]):
    logger.info(f"{name}: {value}")
    # TODO: can we figure out context so we don't have to pass it.
    # What kedro node am I running? Can I set this as my "context" variable
    mlflow.log_metric(f"{context}/{name}", value)
