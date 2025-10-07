import logging

import pandas as pd
import pyspark.sql as ps
from matrix_inject.inject import inject_object

from .evaluations import ComparisonEvaluation

logger = logging.getLogger(__name__)


@inject_object()
def run_evaluation(
    evaluation: ComparisonEvaluation,
    *matrices: ps.DataFrame,
) -> pd.DataFrame:
    """Function to apply evaluation."""
    logger.info(f"Evaluation is: {evaluation}")
    return evaluation.evaluate(matrices)
