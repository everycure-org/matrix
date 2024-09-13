"""Inference pipeline's nodes."""
from matrix import settings
from kedro.pipeline import Pipeline, node, pipeline
from typing import List, Type
import numpy as np
import pandas as pd
from . import infer_runners
from ..modelling.nodes import create_model
from ..modelling.model import ModelWrapper
from ..preprocessing.nodes import resolve
from matplotlib.figure import Figure


def infer_type_from_ids(drug_id: list, disease_id: list) -> str:
    """Infers the type based on the presence or absence of drug and disease IDs.

    Args:
        drug_id (list): List containing drug ID(s).
        disease_id (list): List containing disease ID(s).

    Returns:
        str: The inferred type ("inferPerDrug", "inferPerDisease", or "inferPerPair").

    Raises:
        ValueError: If both drug_id and disease_id are empty.
    """
    if isinstance(disease_id[0], np.float64) & isinstance(drug_id[0], np.float64):
        raise ValueError("Need to specify drug, disease, or both")
    elif isinstance(disease_id[0], np.float64):
        return "inferPerDrug"
    elif isinstance(drug_id[0], np.float64):
        return "inferPerDisease"
    else:
        return "inferPerPair"


def resolve_input_sheet(sheet: pd.DataFrame) -> pd.DataFrame:
    """Run inference on disease or list of diseases and drug list, and write to google sheets.

    Args:
        sheet: google sheet from which we take the drug/disease IDs.

    Returns:
        dataframe saved in gsheets.
    """
    # Load drug/disases IDs
    drug_id = sheet["drug_id"]
    disease_id = sheet["disease_id"]

    # Infer the request types
    infer_type = infer_type_from_ids(drug_id, disease_id)
    print(infer_type)
    return infer_type


def run_inference(
    model: ModelWrapper,
    embeddings: pd.DataFrame,
    drug_nodes: pd.DataFrame,
    disease_nodes: pd.DataFrame,
    train_df: pd.DataFrame,
    sheet: pd.DataFrame,
    infer_type: str,
) -> pd.DataFrame:
    """Run inference on disease or list of diseases and drug list, and write to google sheets.

    Args:
        model: trained model that will be used for inference.
        embeddings: embeddings of KG nodes, will be used for inference.
        drug_nodes: synonymized drug nodes.
        disease_nodes: synonymized disease nodes.
        train_df: training dataframe to highlight any drug-disease pairs which were present in the train data.
        sheet: google sheet where we will write the scores.
        infer_type: type of inference to be run.

    Returns:
        scores: treat scores requested written to google sheets
    """
    inferRunner = getattr(infer_runners, infer_type)
    runner = inferRunner(drug_nodes.curie, disease_nodes.curie)

    # set up the runner with model, nodes, diseases and drugs
    runner.ingest_data(embeddings)

    # run inference for drugs/diseases specified in params
    runner.run_inference(model)

    # generate descriptive statistics for the scores
    runner.generate_stats()

    # add metadata and cross check whether the drug-disease pairs inferred are not in training data
    runner.add_metadata(train_df)

    return runner._scores


def visualise_treat_scores(scores: pd.DataFrame, infer_type: str) -> Figure:
    """Create visualisations based on the treat scores and store them in GCS/MLFlow.

    Args:
        scores: treat scores generated during the inference.
        infer_type: type of inference requested.

    Returns:
        figure: figure saved locally and in MLFlow
    """
    inferRunner = getattr(infer_runners, infer_type)
    runner = inferRunner()
    return runner.visualise_scores(scores)
