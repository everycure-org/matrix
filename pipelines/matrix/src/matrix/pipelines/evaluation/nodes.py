"""Module with nodes for evaluation."""
import abc
from typing import Any, List, Dict, Union

from sklearn.impute._base import _BaseImputer

import pandas as pd

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.unpack import unpack_params
from refit.v1.core.make_list_regexable import _extract_elements_in_list


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
    # TODO: Generator might also be a more primitive generator
    # that outputs e.g., names of drug/diseases, hence entity
    # resolution to the graph is required before passing on.

    # TODO: Generator might be more advanced, e.g., it traverses known
    # edges to curate a realistic test dataset.

    return generator.generate(graph, known_pairs)


def make_test_predictions(
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    score_col_name: str = "treat score",
) -> pd.DataFrame:
    """TO DO.

    Args:
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        score_col_name: Probability score column name.

    Returns:
        _description_
    """
    # TODO: !! Collect embedding vectors for pairs without them

    # Apply transformers to data
    transformed = apply_transformers(data, transformers)

    # TODO: Or shall we split in 2 nodes? Yes probably for when we do looped predictions.
    features = _extract_elements_in_list(transformed.columns, features, raise_exc=True)

    # Generate model probability scores
    transformed[score_col_name] = model.predict_proba(transformed[features].values)[
        :, 1
    ]
    return transformed


# @has_schema(
#     schema={
#         "y": "object",
#         "y_pred": "object",
#     },
#     allow_subset=True,
# )
@inject_object()
def evaluate_test_predictions(data: pd.DataFrame, evaluation: Evaluation) -> Any:
    """Function to apply evaluation.

    Args:
        data: predictions to evaluate on
        evaluation: metric to evaluate
    Returns:
        Evaluation report
    """
    return evaluation.evaluate(data)


def consolidate_reports(*reports) -> dict:
    """Function to consolidate reports into master report.

    Args:
        reports: tuples of (name, report) pairs.

    Returns:
        Dictionary representing consolidated report.
    """
    return [*reports]
