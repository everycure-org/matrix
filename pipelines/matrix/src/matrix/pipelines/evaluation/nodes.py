"""Module with nodes for evaluation."""
import json
from typing import Any, List, Dict, Union, Tuple

from sklearn.impute._base import _BaseImputer

import pandas as pd

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.unpack import unpack_params
from refit.v1.core.make_list_regexable import _extract_elements_in_list

from matrix import settings
from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import DrugDiseasePairGenerator

from matrix.pipelines.modelling.nodes import apply_transformers
from matrix.pipelines.evaluation.evaluation import Evaluation
from matrix.pipelines.modelling.model import ModelWrapper


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
    graph: KnowledgeGraph,
    known_pairs: pd.DataFrame,
    generator: DrugDiseasePairGenerator,
) -> pd.DataFrame:
    """Function to generate test dataset.

    Function leverages the given strategy to construct
    pairs dataset.

    Args:
        graph: KnowledgeGraph instance
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        generator: Generator strategy
    Returns:
        Pairs dataframe
    """
    return generator.generate(graph, known_pairs)


def make_test_predictions(
    graph: KnowledgeGraph,
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    score_col_name: str,
) -> pd.DataFrame:
    """Generated probability scores for drug-disease dataset.

    TODO: Parallelise for large datasets

    Args:
        graph: Knowledge graph.
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        score_col_name: Probability score column name.

    Returns:
        Pairs dataset with additional column containing the probability scores.
    """
    # Collect embedding vectors
    print(
        f"Memory usage before collecting embeddings: {data.memory_usage().sum() / 1e6:.2f} MB"
    )
    data["source_embedding"] = data.apply(
        lambda row: graph._embeddings[row.source], axis=1
    )
    data["target_embedding"] = data.apply(
        lambda row: graph._embeddings[row.target], axis=1
    )
    print(
        f"Memory usage after collecting embeddings: {data.memory_usage().sum() / 1e6:.2f} MB"
    )

    # Apply transformers to data
    print("Applying transformers...")
    transformed = apply_transformers(data, transformers)
    print(
        f"Memory usage after applying transformers: {transformed.memory_usage().sum() / 1e6:.2f} MB"
    )
    features = _extract_elements_in_list(transformed.columns, features, raise_exc=True)
    print(f"Number of features: {len(features)}")

    # Generate model probability scores
    print("Generating model probability scores...")
    transformed[score_col_name] = model.predict_proba(transformed[features].values)[
        :, 1
    ]
    print(
        f"Memory usage after generating scores: {transformed.memory_usage().sum() / 1e6:.2f} MB"
    )
    return transformed


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
