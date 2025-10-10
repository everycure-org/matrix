import logging
from dataclasses import asdict

import matplotlib.pyplot as plt
import polars as pl
from matrix_inject.inject import inject_object

from .evaluations import ComparisonEvaluation
from .input_paths import InputPathsMultiFold

logger = logging.getLogger(__name__)


@inject_object()
def create_input_matrices_dataset(
    input_paths: list[InputPathsMultiFold],
) -> list[dict]:
    """Function to create input matrices dataset."""
    # Return initialised dataclass objects as dictionaries
    return [asdict(v) for v in input_paths]


@inject_object()
def run_evaluation(
    perform_multifold: bool,
    perform_bootstrap: bool,
    evaluation: ComparisonEvaluation,
    input_matrices: dict[str, dict[str, pl.LazyFrame]],
) -> pl.DataFrame:
    """Function to apply evaluation."""
    logger.info(f"Evaluation is: {evaluation}")

    if perform_multifold:
        if perform_bootstrap:
            return evaluation.evaluate_bootstrap_multi_fold(input_matrices)
        else:
            return evaluation.evaluate_multi_fold(input_matrices)
    else:
        if perform_bootstrap:
            return evaluation.evaluate_bootstrap_single_fold(input_matrices)
        else:
            return evaluation.evaluate_single_fold(input_matrices)


@inject_object()
def plot_results(
    perform_multifold: bool,
    perform_bootstrap: bool,
    evaluation: ComparisonEvaluation,
    results: pl.DataFrame,
    input_matrices: dict[str, any],
) -> plt.Figure:
    """Function to plot results."""
    is_plot_errors = perform_multifold or perform_bootstrap
    return evaluation.plot_results(results, input_matrices, is_plot_errors)
