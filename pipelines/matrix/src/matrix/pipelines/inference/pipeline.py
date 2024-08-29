"""Fabricator pipeline."""
from matrix import settings

from kedro.pipeline import Pipeline, node, pipeline


def run_inference(model, df, list):
    # TODO: You can implement inference here
    df["score"] = "test"

    breakpoint()

    return df


def create_pipeline(**kwargs) -> Pipeline:
    """Create fabricator pipeline."""
    nodes = []
    for model in settings.DYNAMIC_PIPELINES_MAPPING.get("modelling"):
        nodes.append(
            node(
                func=run_inference,
                inputs=[
                    f'modelling.{model["model_name"]}.models.model',
                    "raw.inputs",
                    "raw.evaluation.disease_list",
                ],
                outputs=f'model_outputs.{model["model_name"]}.predictions',
                name=f'run_{model["model_name"]}_inference',
            )
        )

    return pipeline(nodes)
