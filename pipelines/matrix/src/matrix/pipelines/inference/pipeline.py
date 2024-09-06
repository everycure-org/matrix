"""Fabricator pipeline."""
from matrix import settings
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create fabricator pipeline."""
    nodes = []
    for model in settings.DYNAMIC_PIPELINES_MAPPING.get("modelling"):
        nodes.append(
            node(
                func=nodes.resolve_input,
                inputs={
                    "sheet": "raw.inputs",
                    "diseases": "raw.evaluation.disease_list",
                    "drugs": "raw.evaluation.drug_list",
                },
                outputs=[
                    "inference.nodes.drugs",
                    "inference.nodes.diseases",
                    "inference.nodes.type",
                ],
                name=f"synonymize",
            ),
            node(
                func=nodes.run_inference,
                inputs={
                    "model": "modelling.xgb.ensemble.models.model",
                    "embeddings": "embeddings.feat.nodes",
                    "infer_type": "inference.nodes.type",
                    "drug_nodes": "inference.nodes.drugs",
                    "disease_nodes": "inference.nodes.diseases",
                    "train_df": "modelling.model_input.splits",  # need it to cross-check if drug-disease pairs we inferred are not in the train set
                    "sheet": "raw.inputs",
                },
                outputs=f"model_outputs.node.predictions",
                name=f"run_inference",
            ),
            node(
                func=nodes.visualise_treat_scores,
                inputs={"scores": "model_outputs.node.predictions"},
                outputs="model_outputs.node.visualisations",
                name=f"visualise_inference",
            ),
        )
    return pipeline(nodes)
