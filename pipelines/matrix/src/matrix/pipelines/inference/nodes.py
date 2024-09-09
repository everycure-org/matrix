"""Inference pipeline's nodes."""
from matrix import settings
from kedro.pipeline import Pipeline, node, pipeline
import numpy as np
import pandas as pd
from . import infer_runners
from ..modelling.nodes import create_model


def synonymize(node_ids):
    """WORK IN PROGRESS. Dummy function, to be populated for synonymization.

    Args:
        node_ids: nodes to synonymize.
    """
    return pd.DataFrame({"id": node_ids})


def resolve_input(sheet, diseases_list, drugs_list):
    """WORK IN PROGRESS. Run inference on disease or list of diseases and drug list, and write to google sheets.

    Args:
        sheet: google sheet from which we take the drug/disease IDs.
        diseases_list: list of all diseases to predict against in case of a drug-centric request.
        drugs_list: list of all diseases to predict against in case of a disease-centric request.
    """
    # TODO: link synonymizer
    drug_id = sheet["drug_id"]
    disease_id = sheet["disease_id"]
    if (len(drug_id[0]) + len(disease_id[0])) == 0:
        raise ValueError("Need to specify drug, disease or both")
    elif len(disease_id[0]) == 0:
        drug_nodes = synonymize(drug_id.values)
        disease_nodes = synonymize(diseases_list.category_class.values)
        infer_type = "inferPerDrug"
    elif len(drug_id[0]) == 0:
        drug_nodes = synonymize(drugs_list.single_ID.values)
        disease_nodes = synonymize(disease_id.values)
        infer_type = "inferPerDisease"
    else:
        drug_nodes = synonymize(drug_id.drug_id.values)
        disease_nodes = synonymize(disease_id.values)
        infer_type = "inferPerPair"
    print(infer_type)
    print(drug_nodes)
    print(disease_nodes)
    return drug_nodes, disease_nodes, infer_type


def run_inference(
    model, embeddings, drug_nodes, disease_nodes, train_df, sheet, infer_type
):
    """Run inference on disease or list of diseases and drug list, and write to google sheets.

    Args:
        model: trained model that will be used for inference.
        embeddings: embeddings of KG nodes, will be used for inference.
        drug_nodes: synonymized drug nodes.
        disease_nodes: synonymized disease nodes.
        train_df: training dataframe to highlight any drug-disease pairs which were present in the train data.
        sheet: google sheet where we will write the scores.
        infer_type: type of inference to be run.
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


def visualise_treat_scores(scores, infer_type):
    """Create visualisations based on the treat scores and store them in GCS/MLFlow."""
    inferRunner = getattr(infer_runners, infer_type)
    runner = inferRunner()
    return runner.visualise_scores(scores)
