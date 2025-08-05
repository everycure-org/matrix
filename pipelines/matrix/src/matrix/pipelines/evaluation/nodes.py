import json
import logging
from typing import Any, Dict, List

import pandas as pd
from matrix.datasets.pair_generator import DrugDiseasePairGenerator
from matrix.inject import inject_object
from matrix.pipelines.evaluation.evaluation import Evaluation
from matrix_schema.utils.pandera_utils import Column, DataFrameSchema, check_output

logger = logging.getLogger(__name__)


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


@check_output(
    schema=DataFrameSchema(
        columns={
            "source": Column(str, nullable=False),
            "target": Column(str, nullable=False),
            "y": Column(int, nullable=False),
        },
        unique=["source", "target"],
    )
)
@inject_object()
def generate_test_dataset(
    known_pairs: pd.DataFrame, matrix: pd.DataFrame, generator: DrugDiseasePairGenerator, score_col_name: str
) -> pd.DataFrame:
    """Function to generate test dataset.

    Function leverages the given strategy to construct
    pairs dataset.

    Args:
        known_pairs: Dataframe containing known pairs with test/train split.
        matrix: Pairs dataframe representing the full matrix with treat scores.
        generator: Generator strategy.
        score_col_name: name of column containing treat scores

    Returns:
        Pairs dataframe
    """

    # Perform checks
    # NOTE: We're currently repeat it for each fold, should
    # we consider moving to matrix outputs?
    check_no_train(matrix, known_pairs)
    check_ordered(matrix, score_col_name)

    return generator.generate(matrix)


@inject_object()
def evaluate_test_predictions(data: pd.DataFrame, evaluation: Evaluation, score_col_name: str) -> Any:
    """Function to apply evaluation.

    Args:
        data: predictions to evaluate on
        evaluation: metric to evaluate.
        score_col_name: name of the score column to use

    Returns:
        Evaluation report
    """
    logger.info(f"Evaluation data size: {data.shape}")
    logger.info(f"Evaluation is: {evaluation}")
    return evaluation.evaluate(data, score_col_name)


@inject_object()
def aggregate_metrics(aggregation_functions: List[Dict], *metrics) -> Dict:
    """
    Aggregate metrics for the separate folds into a single set of metrics.

    Args:
        aggregation_functions: List of dictionaries containing the name and object of the aggregation function.
        metrics: Dictionaries of metrics for all folds.
    """
    # Extract list of metrics for each fold and check consistency
    metric_names_lst_all_folds = [list(report.keys()) for report in metrics]
    metric_names_lst = metric_names_lst_all_folds[0]
    if not all(metric_names == metric_names_lst_all_folds[0] for metric_names in metric_names_lst_all_folds):
        raise ValueError("Inconsistent metrics across folds. Each fold should have the same set of metrics.")

    # Perform aggregation
    aggregated_metrics = dict()
    for agg_func in aggregation_functions:
        aggregated_metrics[agg_func.__name__] = {
            metric_name: agg_func([report[metric_name] for report in metrics]) for metric_name in metric_names_lst
        }

    return json.loads(json.dumps(aggregated_metrics, default=float))


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

    def add_report(master_report: dict, evaluation: str, type: str, report: dict) -> dict:
        """Add a metrics to the master report, appending the evaluation type to the metric name.

        Args:
            master_report: Master report to add to.
            evaluation: Evaluation name.
            type: Type of report (e.g. mean, fold_1,...).
            report: Report to add.
        """

        for metric, value in report.items():
            # Add evaluation type suffix to the metric name
            full_metric_name = evaluation + "_" + metric

            # Add keys for metrics name and type if not present
            if full_metric_name not in master_report:
                master_report[full_metric_name] = {}
            if type not in master_report[full_metric_name]:
                master_report[full_metric_name][type] = {}

            # Add value to the metric name and type
            master_report[full_metric_name][type] = value

        return master_report

    master_report = {}
    for report_name, report in reports.items():
        # Parse the report name key created in evaluation/pipeline.py
        evaluation, fold_or_aggregated = report_name.split(".")

        # In the case of aggregated results, add the results for each aggregation function
        if fold_or_aggregated == "aggregated":
            for aggregation, report in report.items():
                master_report = add_report(master_report, evaluation, aggregation, report)

        # In the case of a fold result, add the results directly for the fold
        else:
            fold = fold_or_aggregated
            master_report = add_report(master_report, evaluation, fold, report)

    return json.loads(json.dumps(master_report, default=float))


@inject_object()
def evaluate_stability_predictions(
    overlapping_pairs: pd.DataFrame, evaluation: Evaluation, *matrices: pd.DataFrame, score_col_name: str
) -> Any:
    """Function to apply stability evaluation.

    Args:
        overlapping_pairs: pairs that overlap across all matrices.
        evaluation: stability metric to use for evaluation.
        matrix_1: full matrix coming from one model
        matrix_2: full matrix coming from another model to compare against
    Returns:
        Evaluation report
    """
    return evaluation.evaluate(overlapping_pairs, matrices, score_col_name)


@inject_object()
def generate_overlapping_dataset(
    generator: DrugDiseasePairGenerator, *matrices: pd.DataFrame, score_col_name: str
) -> pd.DataFrame:
    """Function to generate overlapping dataset.

    Args:
        generator: generator strategy
        matrices: DataFrames coming from different models to compare against
        score_col_name: name of the score column to use
    Returns:
        Evaluation report
    """
    return generator.generate(matrices, score_col_name)


def calculate_rank_commonality(ranking_output: dict, commonality_output: dict) -> Any:
    """Function to calculate rank commonality (custom metric).

    Args:
        ranking_output: ranking output
        commonality_output: commonality output

    Returns:
        rank commonality output
    """
    # clean MLFlow names; if running in base, the names remain unchanged
    commonality_output = {k.split("stability_overlap_")[-1]: v for k, v in commonality_output.items()}
    ranking_output = {k.split("stability_ranking_")[-1]: v for k, v in ranking_output.items()}
    # calculate
    rank_commonality_output = {}
    ranking_output = {k: v for k, v in ranking_output.items() if ("spearman" in k)}
    n_ranking_values = [int(n.split("_")[-2]) for n in ranking_output.keys() if "p_value" not in n]
    n_commonality_values = [int(n.split("_")[-1]) for n in commonality_output.keys() if n.split("_")[-1]]
    n_values = list(set(n_ranking_values) & set(n_commonality_values))
    # Compute harmonic mean between Commonality@n and Spearman-rank@n
    for i in n_values:
        # Spearman correlation is between -1 and 1, taking the absolute value to avoid division by small numbers
        r_k = abs(ranking_output[f"spearman_at_{i}_stat"])
        c_k = commonality_output[f"commonality_at_{i}"]
        if r_k + c_k == 0:
            s_f1 = None
        elif pd.isnull(r_k) | pd.isnull(c_k):
            s_f1 = None
        else:
            s_f1 = (2 * r_k * c_k) / (r_k + c_k)
        rank_commonality_output[f"rank_commonality_at_{i}_stat"] = s_f1
        rank_commonality_output[f"rank_commonality_at_{i}_pvalue"] = ranking_output[f"spearman_at_{i}_pvalue"]
    return json.loads(json.dumps(rank_commonality_output, default=float))
