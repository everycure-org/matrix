import logging
from typing import List

import pandas as pd
import pyspark.sql as ps
from matrix_inject.inject import inject_object

from .evaluations import ComparisonEvaluation

logger = logging.getLogger(__name__)


@inject_object()
def run_evaluation(*matrices: List[ps.DataFrame], evaluation_object: ComparisonEvaluation) -> pd.DataFrame:
    """Function to apply evaluation.

    Args:
        data: predictions to evaluate on
        evaluation: metric to evaluate.
        score_col_name: name of the score column to use

    Returns:
        Evaluation report
    """
    # logger.info(f"Evaluation data size: {data.shape}")
    logger.info(f"Evaluation is: {evaluation_object}")
    return evaluation_object.evaluate(matrices)
