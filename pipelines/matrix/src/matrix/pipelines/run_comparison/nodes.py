import logging
from dataclasses import asdict

import polars as pl
from matrix_inject.inject import inject_object

from .evaluations import ComparisonEvaluation
from .input_paths import InputPathsMultiFold

logger = logging.getLogger(__name__)


@inject_object()
def create_input_matrices_dataset(
    input_paths: dict[str, InputPathsMultiFold],
) -> InputPathsMultiFold:
    """Function to create input matrices dataset."""
    # Return initialised dataclass objects as dictionaries
    return {k: asdict(v) for k, v in input_paths.items()}


@inject_object()
def run_evaluation(
    input_matrices: dict[str, dict[str, pl.LazyFrame]],
    evaluation: ComparisonEvaluation,
    input_paths: InputPathsMultiFold,
) -> pl.DataFrame:
    """Function to apply evaluation."""
    logger.info(f"Evaluation is: {evaluation}")
    return evaluation.evaluate(input_matrices)
