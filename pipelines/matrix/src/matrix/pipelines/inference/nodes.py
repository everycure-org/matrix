"""Inference pipeline's nodes."""
from matrix import settings
from kedro.pipeline import Pipeline, node, pipeline
from typing import List, Type
import numpy as np
import pandas as pd

from . import visualisers
from ..modelling.model import ModelWrapper
from matplotlib.figure import Figure


def resolve_input_sheet(
    input_sheet: pd.DataFrame, drug_sheet: pd.DataFrame, disease_sheet: pd.DataFrame
) -> pd.DataFrame:
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
    # Load drug/disases IDs
    drug_id = input_sheet["drug_id"]
    disease_id = input_sheet["disease_id"]

    # Rule-based check of the request type
    if isinstance(disease_id[0], np.float64) & isinstance(drug_id[0], np.float64):
        raise ValueError("Need to specify drug, disease, or both")
    elif isinstance(disease_id[0], np.float64):
        drug_list = pd.DataFrame({"curie": drug_id})
        disease_list = disease_sheet
        return "Drug-centric predictions", drug_list, disease_list
    elif isinstance(drug_id[0], np.float64):
        drug_list = drug_sheet
        disease_list = pd.DataFrame({"curie": disease_id})
        return "Disease-centric predictions", drug_list, disease_list
    else:
        drug_list = pd.DataFrame({"curie": drug_id})
        disease_list = pd.DataFrame({"curie": disease_id})
        return "Drug-disease specific predictions", drug_list, disease_list


def visualise_treat_scores(scores: pd.DataFrame, infer_type: str) -> Figure:
    """Create visualisations based on the treat scores and store them in GCS/MLFlow.

    Args:
        scores: treat scores generated during the inference.
        infer_type: type of inference requested.

    Returns:
        figure: figure saved locally and in MLFlow
    """
    # FUTURE: add more visualisations
    kde_plot = visualisers.create_kdeplot(scores, infer_type)
    return kde_plot
