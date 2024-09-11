"""Module with nodes for evaluation."""
import json
from tqdm import tqdm
from typing import Any, List, Dict, Union

from sklearn.impute._base import _BaseImputer

import pandas as pd

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key
from refit.v1.core.unpack import unpack_params
from refit.v1.core.make_list_regexable import _extract_elements_in_list

from matrix import settings
from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import DrugDiseasePairGenerator

from matrix.pipelines.matrix_generation.nodes import make_batch_predictions
from matrix.pipelines.evaluation.evaluation import Evaluation
from matrix.pipelines.modelling.model import ModelWrapper


def create_prm_clinical_trials(raw_clinical_trials: pd.DataFrame) -> pd.DataFrame:
    """Function to clean clinical trails dataset.

    Args:
        raw_clinical_trials: Raw clinical trails data
    Returns:
        Cleaned clinical trails data
    """
    clinical_trail_data = raw_clinical_trials.rename(
        columns={"drug_kg_curie": "source", "disease_kg_curie": "target"}
    )

    # Create the 'y' column where y=1 means 'significantly_better' and y=0 means 'significantly_worse'
    clinical_trail_data["y"] = clinical_trail_data.apply(
        lambda row: 1
        if row["significantly_better"] == 1
        else (0 if row["significantly_worse"] == 1 else None),
        axis=1,
    )

    # Remove rows where 'y' is None (which means they are not 'significantly_better' or 'significantly_worse')
    clinical_trail_data = clinical_trail_data.dropna(subset=["y"])

    # Use columns 'source', 'target', and 'y' only
    clinical_trail_data = (
        clinical_trail_data[["source", "target", "y"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return clinical_trail_data


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
    clinical_trials_data: pd.DataFrame,
) -> pd.DataFrame:
    """Function to generate test dataset.

    Function leverages the given strategy to construct
    pairs dataset.

    Args:
        graph: KnowledgeGraph instance
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        generator: Generator strategy
        clinical_trials_data: clinical trails data
    Returns:
        Pairs dataframe
    """
    return generator.generate(
        graph, known_pairs, clinical_trials_data=clinical_trials_data
    )


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
