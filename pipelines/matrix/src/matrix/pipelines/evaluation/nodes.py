"""Module with nodes for evaluation."""
import abc
from typing import Any, List, Dict, Union

from sklearn.impute._base import _BaseImputer

import pandas as pd

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.unpack import unpack_params
from refit.v1.core.make_list_regexable import _extract_elements_in_list


from matrix.datasets.graph import KnowledgeGraph, DrugDiseasePairGenerator

from matrix.pipelines.modelling.nodes import apply_transformers, get_model_predictions
from matrix.pipelines.modelling.model import ModelWrapper


@inject_object()
def generate_test_dataset(
    graph: KnowledgeGraph,
    generator: DrugDiseasePairGenerator,
) -> pd.DataFrame:
    """Function to generate test dataset.

    Function leverages the given strategy to construct
    pairs dataset.

    Args:
        graph: KnowledgeGraph instance
        generator: Generator strategy
    Returns:
        Pairs dataframe
    """
    # TODO: Generator might also be a more primitive generator
    # that outputs e.g., names of drug/diseases, hence entity
    # resolution to the graph is required before passing on.

    # TODO: Generator might be more advanced, e.g., it traverses known
    # edges to curate a realistic test dataset.

    return generator.generate(
        graph, pd.DataFrame([], columns=["source", "target", "y", "split"])
    )


def make_test_predictions(
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    target_col_name: str,
    prediction_suffix: str = "_pred",
) -> pd.DataFrame:
    """TO DO.

    Args:
        data: _description_
        transformers: _description_
        model: _description_
        features: _description_
        target_col_name: _description_
        prediction_suffix: _description_. Defaults to "_pred".

    Returns:
        _description_
    """
    # Apply transformers to data
    transformed = apply_transformers(data, transformers)

    # TODO: Or shall we split in 2 nodes?
    features = _extract_elements_in_list(transformed.columns, features, raise_exc=True)

    # Make model predictions
    return get_model_predictions(transformed, model, features, target_col_name)


# # TODO: Will have to move to other file
# class Evaluation(abc.ABC):
#     # Does this function need anything else to operate?
#     def evaluate(self, data: pd.DataFrame):
#         ...


# class MRREvaluation(Evaluation):
#     # Does this function need anything else to operate?
#     def evaluate(self, data: pd.DataFrame):
#         # TODO: Implement here
#         return {"evaluation": "hitk"}


# class HitKEvaluation(Evaluation):
#     # Does this function need anything else to operate?
#     def evaluate(self, data: pd.DataFrame):
#         # TODO: Implement here
#         return {"evaluation": "hitk"}


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
