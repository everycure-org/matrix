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
    """Function to apply evaluation.

    Args:
        data: predictions to evaluate on
        evaluation: metric to evaluate.
        score_col_name: name of the score column to use

    Returns:
        Evaluation report
    """
    logger.info(f"Evaluation is: {evaluation}")
    return evaluation.evaluate(matrices)
