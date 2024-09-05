"""Inference pipeline's nodes."""
from matrix import settings
from kedro.pipeline import Pipeline, node, pipeline

from infer_runners import *


def run_inference(model, nodes, train_df, sheet, diseases, drugs, runner):
    """Run inference on disease or list of diseases and drug list, and write to google sheets.

    Args:
        model: trained model that will be used for inference
        nodes: embeddings of KG nodes, will be used for inference
        train_df: training dataframe to highlight any drug-disease pairs which were present in the train data
        sheet: google sheet where we will write the scores
        diseases: disease list
        drugs: drugs list
        runner: wrapper for different inference requests
    """
    # set up the runner with model, nodes, diseases and drugs
    runner.ingest_data(nodes, diseases, drugs)

    # run inference for drugs/diseases specified in params
    runner.run_inference(model)

    # cross check whether the drug-disease pairs inferred are not in training data
    runner.cross_check(train_df)

    # generate descriptive statistics for the scores
    runner.generate_stats()

    return runner.scores


def visualise_treat_scores(runner):
    """Create visualisations based on the treat scores and store them in GCS/MLFlow."""
    return runner.visualise_scores()
