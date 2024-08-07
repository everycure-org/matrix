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
from matrix.pipelines.evaluation.node_synonymizer.node_synonymizer import NodeSynonymizer
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
    data["source_embedding"] = data.apply(
        lambda row: graph._embeddings[row.source], axis=1
    )
    data["target_embedding"] = data.apply(
        lambda row: graph._embeddings[row.target], axis=1
    )

    # Apply transformers to data
    transformed = apply_transformers(data, transformers)
    features = _extract_elements_in_list(transformed.columns, features, raise_exc=True)

    # Generate model probability scores
    transformed[score_col_name] = model.predict_proba(transformed[features].values)[
        :, 1
    ]
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


def clean_clinical_trial_data(clinical_trial_data: pd.DataFrame, node_synonymizer_db_path: str) -> pd.DataFrame:
    """Function to clean clinical trials dataset for use in time-split evaluation metrics.
    
       Clinical trial data should be a EXCEL format containg 8 columns: 
       Clinical Trial #, Reason for Rejection, Mapped Drug(s), Mapped Disease, Significantly Better?, Non-Significantly Better?, Non-Significantly Worse?, Significantly Worse?
       
       1. We filter out rows with the following conditions:
            - Missing Mapped Drug(s) name
            - Missing Mapped Disease name
            - with reason for rejection
            - missing values in either significantly better, non-significantly better, non-significantly worse, or significantly worse columns
            
       2. Map the disease_id and drug_id to the corresponding curie ids and filter out rows without mapping ids.
            - add column drug_kg_id
            - add column disease_kg_id
       
       3. Standardize the column names to the following:
            - Mapped Drug(s) name -> drug_name
            - Mapped Disease -> disease_name
            - Significantly Better? -> significantly_better
            - Non-Significantly Better? -> non_significantly_better
            - Non-Significantly Worse? -> non_significantly_worse
            - Significantly Worse? -> significantly_worse
         
       4. Only keep the following columns:
            - drug_name
            - disease_name
            - drug_kg_id
            - disease_kg_id
            - significantly_better
            - non_significantly_better
            - non_significantly_worse
            - significantly_worse

    Args:
        clinical_trial_data: Clinical trial data provided by medical team for evaluation.

    Returns:
        Cleaned clinical trial data.
    """
    
    ## connect to node_synonymizer database
    node_synonymizer = NodeSynonymizer(node_synonymizer_db_path)
    
    # rename all columns to make consistent
    clinical_trial_data.columns = ['clinical_trial_id', 'reason_for_rejection', \
                                   'drug_name', 'disease_name', 'significantly_better', \
                                   'non_significantly_better', 'non_significantly_worse', 'significantly_worse']

    # remove rows with reason for rejection
    clinical_trial_data = clinical_trial_data[clinical_trial_data['reason_for_rejection'].map(lambda x: type(x) != str)].reset_index(drop=True)
    
    # remove rows with missing drug or disease name
    row_has_missing = clinical_trial_data['drug_name'].isna() | clinical_trial_data['disease_name'].isna()
    clinical_trial_data = clinical_trial_data[~row_has_missing].reset_index(drop=True)
    
    # remove rows with missing values in significantly better, non-significantly better, non-significantly worse, or significantly worse columns
    row_has_missing = clinical_trial_data['significantly_better'].isna() | clinical_trial_data['non_significantly_better'].isna() | \
                        clinical_trial_data['non_significantly_worse'].isna() | clinical_trial_data['significantly_worse'].isna()
    clinical_trial_data = clinical_trial_data[~row_has_missing].reset_index(drop=True)
    
    # drop columns: clinical_trial_id, reason_for_rejection
    clinical_trial_data = clinical_trial_data.drop(columns=['clinical_trial_id', 'reason_for_rejection'])
    
    
    
    return clinical_trial_data.dropna(subset=["disease_id", "drug_id"])