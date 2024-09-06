"""Inference pipeline's nodes."""
# from matrix import settings
# from kedro.pipeline import Pipeline, node, pipeline

# from infer_runners import *


# def synonymize(node_ids):
#     """WORK IN PROGRESS. Dummy function, to be populated for synonymization.

#     Args:
#         node_ids: nodes to synonymize.
#     """
#     return node_ids


# def resolve_input(sheet, diseases_list, drugs_list):
#     """WORK IN PROGRESS. Run inference on disease or list of diseases and drug list, and write to google sheets.

#     Args:
#         sheet: google sheet from which we take the drug/disease IDs.
#         diseases_list: list of all diseases to predict against in case of a drug-centric request.
#         drug_list: list of all diseases to predict against in case of a disease-centric request.
#     """
#     drug_id = sheet["drug_id"]
#     disease_id = sheet["drug_id"]
#     if (drug_id is None) & (disease is None):
#         raise ValueError("Need to specify drug, disease or both")
#     elif disease_id is None:
#         drug_nodes = synonymize(drug_id)
#         disease_nodes = synonymize(diseases_list)
#         infer_type = "inferPerPair"
#     elif drug_id is None:
#         drug_nodes = synonymize(drugs_list)
#         disease_nodes = synonymize(disease_id)
#         infer_type = "inferPerDrug"
#     else:
#         drug_nodes = synonymize(drug_id)
#         disease_nodes = synonymize(disease_id)
#         infer_type = "inferPerDisease"
#     return drug_nodes, disease_nodes, infer_type


# def run_inference(
#     model, embeddings, drug_nodes, disease_nodes, train_df, sheet, infer_type
# ):
#     """Run inference on disease or list of diseases and drug list, and write to google sheets.

#     Args:
#         model: trained model that will be used for inference.
#         embeddings: embeddings of KG nodes, will be used for inference.
#         query_nodes: synonymized nodes.
#         train_df: training dataframe to highlight any drug-disease pairs which were present in the train data.
#         sheet: google sheet where we will write the scores.
#         diseases: disease list.
#         drugs: drugs list.
#         runner: wrapper for different inference requests.
#     """
#     inferRunner = getattr(infer_runners, infer_type)
#     runner = inferRunner()

#     # set up the runner with model, nodes, diseases and drugs
#     runner.ingest_data(embeddings, drug_nodes, disease_nodes)

#     # run inference for drugs/diseases specified in params
#     runner.run_inference(model, sheet)

#     # cross check whether the drug-disease pairs inferred are not in training data
#     runner.cross_check(train_df)

#     # generate descriptive statistics for the scores
#     runner.generate_stats()

#     return runner.scores


# def visualise_treat_scores(runner):
#     """Create visualisations based on the treat scores and store them in GCS/MLFlow."""
#     return runner.visualise_scores()
