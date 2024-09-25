"""Module with nodes for evaluation."""
import json
from typing import Any, List, Dict, Union

from sklearn.impute._base import _BaseImputer

import pandas as pd

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema

from matrix import settings
from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import DrugDiseasePairGenerator

from matrix.pipelines.matrix_generation.nodes import make_batch_predictions
from matrix.pipelines.evaluation.evaluation import Evaluation
from matrix.pipelines.modelling.model import ModelWrapper


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
        raise ValueError(
            f"Found {len(overlapping_pairs)} pairs in test set that also appear in training set."
        )


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


def make_test_predictions(
    graph: KnowledgeGraph,
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    score_col_name: str,
    batch_by: str = "target",
) -> pd.DataFrame:
    """Generate probability scores for drug-disease dataset.

    Args:
        graph: Knowledge graph.
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        score_col_name: Probability score column name.
        batch_by: Column to use for batching (e.g., "target" or "source").

    Returns:
        Pairs dataset with additional column containing the probability scores.
    """
    return make_batch_predictions(
        graph, data, transformers, model, features, score_col_name, batch_by=batch_by
    )


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
        for idx_2, evaluation in enumerate(
            settings.DYNAMIC_PIPELINES_MAPPING.get("evaluation")
        ):
            master_report[model["model_name"]][
                evaluation["evaluation_name"]
            ] = reports_lst[idx_1 + idx_2]
    return json.loads(json.dumps(master_report, default=float))
