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
                func=nodes.run_inference,
                inputs={
                    "model": f'modelling.{model["model_name"]}.models.model',
                    "nodes": "embeddings.feat.nodes",
                    "train_df": "modelling.model_input.splits",  # need it to cross-check if drug-disease pairs we inferred are not in the train set
                    "sheet": "raw.inputs",
                    "diseases": "raw.evaluation.disease_list",
                    "drugs": "raw.evaluation.drug_list",
                    "runner": "params:inference.runner",  # class injected with object_injector, suitable for different inference requests
                },
                outputs=f'model_outputs.{model["model_name"]}.predictions',
                name=f'run_{model["model_name"]}_inference',
            ),
            node(
                func=nodes.visualise_treat_scores,
                inputs={
                    "scores": f'model_outputs.{model["model_name"]}.predictions',
                    "runner": "params:inference.runner",
                },
                outputs=f'model_outputs.{model["model_name"]}.treat_score_visualisations',
                name=f'visualise_{model["model_name"]}_inference',
            ),
        )
    return pipeline(nodes)
