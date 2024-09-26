"""Inference pipeline's nodes."""
from matrix import settings
from kedro.pipeline import Pipeline, node, pipeline
from typing import List, Type, Dict
import numpy as np
import pandas as pd

from . import visualisers
from ..modelling.model import ModelWrapper
from matplotlib.figure import Figure


def resolve_input_sheet(
    input_sheet: pd.DataFrame, drug_sheet: pd.DataFrame, disease_sheet: pd.DataFrame
) -> (Dict, pd.DataFrame, pd.DataFrame):
    """Run inference on disease or list of diseases and drug list, and write to google sheets.

    Args:
        input_sheet: dataframe with users' input IDs.
        drug_sheet: drug list with normalized curies.
        disease_sheet: disease list with normalized curies.

    Returns:
        str: The inferred type ("inferPerDrug", "inferPerDisease", or "inferPerPair").

    Raises:
        ValueError: If both drug_id and disease_id are empty.
    """
    # Load drug/disases IDs and timestamp from the msot re
    drug_id = input_sheet["norm_drug_id"].values[-1]
    drug_name = input_sheet["norm_drug_name"].values[-1]
    disease_id = input_sheet["norm_disease_id"].values[-1]
    disease_name = input_sheet["norm_disease_name"].values[-1]
    timestamp = input_sheet["timestamp"].values[-1].split(" ")[0]
    # Rule-based check of the request type
    if isinstance(disease_id, float) and isinstance(drug_id, float):
        raise ValueError("Need to specify drug, disease, or both")
    elif isinstance(disease_id, float):
        drug_list = pd.DataFrame({"curie": [drug_id], "name": [drug_name]})
        disease_list = disease_sheet
        request = "Drug-centric predictions"
    elif isinstance(drug_id, float):
        drug_list = drug_sheet
        disease_list = pd.DataFrame({"curie": [disease_id], "name": [disease_name]})
        request = "Disease-centric predictions"
    else:
        drug_list = pd.DataFrame({"curie": [drug_id], "name": [drug_name]})
        disease_list = pd.DataFrame({"curie": [disease_id], "name": [disease_name]})
        request = "Drug-disease specific predictions"
    return {"time": timestamp, "request": request}, drug_list, disease_list


def visualise_treat_scores(
    scores: pd.DataFrame, infer_type: Dict, col_name: str
) -> Figure:
    """Create visualisations based on the treat scores and store them in GCS/MLFlow.

    Args:
        scores: treat scores generated during the inference.
        infer_type: type of inference requested.
        col_name: name of the column with treat scores

    Returns:
        figure: figure saved locally and in MLFlow
    """
    # FUTURE: add more visualisations
    kde_plot = visualisers.create_kdeplot(scores, infer_type, col_name)
    return kde_plot
