import json
from typing import Any


import pandas as pd

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema

from matrix.datasets.pair_generator import DrugDiseasePairGenerator

from matrix.pipelines.evaluation.evaluation import Evaluation


def check_no_train(data: pd.DataFrame, known_pairs: pd.DataFrame) -> None:
    """Checks that no pairs in the ground truth training set appear in the data.

    Args:
        data: Pairs dataset to check.
        known_pairs: DataFrame with known drug-disease pairs.

    Raises:
        ValueError: If any training pairs are found in the data.
    """
    is_test = known_pairs["split"].eq("TEST")
    train_pairs = known_pairs[~is_test]
    train_pairs_set = set(zip(train_pairs["source"], train_pairs["target"]))
    data_pairs_set = set(zip(data["source"], data["target"]))
    overlapping_pairs = data_pairs_set.intersection(train_pairs_set)

    if overlapping_pairs:
        raise ValueError(f"Found {len(overlapping_pairs)} pairs in test set that also appear in training set.")


def check_ordered(
    data: pd.DataFrame,
    score_col_name: str,
) -> None:
    """Check if the score column is correctly ordered.

    Args:
        data: DataFrame containing score column.
        score_col_name: Name of the column containing the scores.

    Raises:
        ValueError: If the score column is not correctly ordered.
    """
    if not data[score_col_name].is_monotonic_decreasing:
        raise ValueError(f"The '{score_col_name}' column is not monotonically descending.")


def perform_matrix_checks(matrix: pd.DataFrame, known_pairs: pd.DataFrame, score_col_name: str) -> None:
    """Perform various checks on the evaluation dataset.

    Args:
        matrix: DataFrame containing a sorted matrix pairs dataset with probability scores, ranks and quantile ranks.
        known_pairs: DataFrame with known drug-disease pairs.
        score_col_name: Name of the column containing the treat scores.

    Raises:
        ValueError: If any of the checks fail.
    """
    check_no_train(matrix, known_pairs)
    check_ordered(matrix, score_col_name)


@has_schema(
    schema={
        "source": "object",
        "target": "object",
        "y": "int",
    },
    allow_subset=True,
)
@inject_object()
def generate_test_dataset(
    matrix: pd.DataFrame,
    generator: DrugDiseasePairGenerator,
) -> pd.DataFrame:
    """Function to generate test dataset.

    Function leverages the given strategy to construct
    pairs dataset.

    Args:
        matrix: Pairs dataframe representing the full matrix with treat scores.
        generator: Generator strategy.

    Returns:
        Pairs dataframe
    """
    return generator.generate(matrix)


@inject_object()
def evaluate_test_predictions(data: pd.DataFrame, evaluation: Evaluation) -> Any:
    """Function to apply evaluation.

    Args:
        data: predictions to evaluate on
        evaluation: metric to evaluate.

    Returns:
        Evaluation report
    """
    return evaluation.evaluate(data)


def reduce_aggregated_results(aggregated_results: dict, aggregation_function_names: list) -> dict:
    """Reduce the aggregated results to a simpler format for MLFlow readout.

    Args:
        aggregated_results: Aggregated results to reduce.
        aggregation_function_names: Names of aggregation functions to report.

    Returns:
        Reduced aggregated results.
    """
    reduced_report = {
        aggregation_name
        + "_"
        + metric_name: metric_value  # concatenate aggregation name and metric name (e.g. mean_mrr)
        for aggregation_name in aggregation_function_names
        for metric_name, metric_value in aggregated_results[aggregation_name].items()
    }
    return json.loads(json.dumps(reduced_report, default=float))


def consolidate_evaluation_reports(**reports) -> dict:
    """Function to consolidate evaluation reports for all models, evaluation types and folds/aggregations into a master report.

    Args:
        reports: tuples of (name, report) pairs.

    Returns:
        Dictionary representing consolidated report.
    """

    def add_report(master_report: dict, model: str, evaluation: str, type: str, report: dict) -> dict:
        """Add a metrics to the master report, appending the evaluation type to the metric name.

        Args:
            master_report: Master report to add to.
            model: Model name.
            evaluation: Evaluation name.
            type: Type of report (e.g. mean, fold_1,...).
            report: Report to add.
        """
        # Add key for model if not present
        if model not in master_report:
            master_report[model] = {}

        for metric, value in report.items():
            # Add evaluation type suffix to the metric name
            full_metric_name = evaluation + "_" + metric

            # Add keys for metrics name and type if not present
            if full_metric_name not in master_report[model]:
                master_report[model][full_metric_name] = {}
            if type not in master_report[model][full_metric_name]:
                master_report[model][full_metric_name][type] = {}

            # Add value to the metric name and type
            master_report[model][full_metric_name][type] = value

        return master_report

    master_report = {}
    for report_name, report in reports.items():
        # Parse the report name key created in evaluation/pipeline.py
        model, evaluation, fold_or_aggregated = report_name.split(".")

        # In the case of aggregated results, add the results for each aggregation function
        if fold_or_aggregated == "aggregated":
            for aggregation, report in report.items():
                master_report = add_report(master_report, model, evaluation, aggregation, report)

        # In the case of a fold result, add the results directly for the fold
        else:
            fold = fold_or_aggregated
            master_report = add_report(master_report, model, evaluation, fold, report)

    return json.loads(json.dumps(master_report, default=float))
