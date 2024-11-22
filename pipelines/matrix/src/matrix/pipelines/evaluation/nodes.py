import json
from typing import Any


import pandas as pd

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema

from matrix import settings
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


def consolidate_evaluation_reports(*reports) -> dict:
    """Function to consolidate evaluation reports into master report.

    Args:
        reports: tuples of (name, report) pairs.

    Returns:
        Dictionary representing consolidated report.
    """
    reports_lst = [*reports]
    master_report = dict()
    for idx_1, model in enumerate(settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")):
        master_report[model["model_name"]] = dict()
        for idx_2, evaluation in enumerate(settings.DYNAMIC_PIPELINES_MAPPING.get("evaluation")):
            master_report[model["model_name"]][evaluation["evaluation_name"]] = reports_lst[idx_1 + idx_2]
    return json.loads(json.dumps(master_report, default=float))
