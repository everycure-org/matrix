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


def resolve_input_sheet(
    sheet: pd.DataFrame,
):
    """Run inference on disease or list of diseases and drug list, and write to google sheets.

    Args:
        sheet: google sheet from which we take the drug/disease IDs..
    """
    drug_id = sheet["drug_id"]
    disease_id = sheet["disease_id"]
    if (len(drug_id[0]) + len(disease_id[0])) == 0:
        raise ValueError("Need to specify drug, disease or both")
    elif len(disease_id[0]) == 0:
        infer_type = "inferPerDrug"
    elif len(drug_id[0]) == 0:
        infer_type = "inferPerDisease"
    else:
        infer_type = "inferPerPair"
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
    runner = inferRunner(drug_nodes, disease_nodes)

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
