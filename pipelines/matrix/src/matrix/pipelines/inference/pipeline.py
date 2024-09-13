"""Fabricator pipeline."""
from matrix import settings
from kedro.pipeline import Pipeline, node, pipeline
from . import nodes as nd


def create_pipeline(**kwargs) -> Pipeline:
    """Create fabricator pipeline."""
    return pipeline(
        [
            node(
                func=lambda x: x,
                inputs="ingestion.raw.drug_list",
                outputs="inference.sheet.drug_list",
                name=f"ingest_drug_list",
            ),
            node(
                func=lambda x: x,
                inputs="ingestion.raw.disease_list",
                outputs="inference.sheet.disease_list",
                name=f"ingest_disease_list",
            ),
            node(
                func=nd.resolve_input_sheet,
                inputs={
                    "sheet": "raw.inputs",
                },
                outputs="inference.nodes.type",
                name=f"choose_infer_type",
            ),
            node(
                func=nd.run_inference,
                inputs={
                    "model": "inference.model.xgb",  # TODO: link params.yaml so that it feeds into this line?
                    "embeddings": "inference.embed.nodes",
                    "infer_type": "inference.nodes.type",
                    "drug_nodes": "ingestion.raw.drug_list:",
                    "disease_nodes": "ingestion.raw.disease_list:",
                    "train_df": "modelling.model_input.splits",
                    "sheet": "raw.inputs",
                },
                outputs=f"model_outputs.node.predictions",
                name=f"run_inference",
            ),
            node(
                func=nd.visualise_treat_scores,
                inputs={
                    "scores": "model_outputs.node.predictions",
                    "infer_type": "inference.nodes.type",
                },
                outputs="model_outputs.node.visualisations",
                name=f"visualise_inference",
            ),
        ]
    )
