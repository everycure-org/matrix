import logging

import pandas as pd
import pyspark.sql as ps
from matrix_inject.inject import inject_object

from .evaluations import ComparisonEvaluation
from .input_paths import InputPathsMultiFold

logger = logging.getLogger(__name__)


@inject_object()
def create_input_matrices_dataset(
    input_paths: dict[str, InputPathsMultiFold],
) -> InputPathsMultiFold:
    """Function to create input matrices dataset."""
    # Return initialised InputPathsMultiFold object, which will be written as MultiMatricesDataset when the node runs in the pipeline.
    # breakpoint()
    return input_paths


@inject_object()
def run_evaluation(
    matrix: ps.DataFrame,
    evaluation: ComparisonEvaluation,
    bool_test_col: str,
    score_col: str,
) -> pd.DataFrame:
    """Function to apply evaluation."""
    logger.info(f"Evaluation is: {evaluation}")
    return evaluation.evaluate(matrix)
