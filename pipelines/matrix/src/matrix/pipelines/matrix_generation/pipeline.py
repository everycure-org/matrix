"""Matrix generation pipeline."""
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from matrix import settings

from . import nodes


def _create_matrix_generation_pipeline(model: str) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.make_predictions_and_sort,
                inputs=[
                    "modelling.feat.rtx_kg2",
                    "matrix_generation.prm.pairs",
                    f"modelling.{model}.model_input.transformers",
                    f"modelling.{model}.models.model",
                    f"params:modelling.{model}.model_options.model_tuning_args.features",
                    f"params:evaluation.score_col_name",
                    "params:matrix_generation.matrix_generation_options.batch_by",
                ],
                outputs=f"matrix_generation.{model}.model_output.sorted_matrix_predictions",
                name=f"make_{model}_predictions_and_sort",
            ),
        ],
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create matrix generation pipeline."""
    initial_node = pipeline(
        [
            node(
                func=nodes.generate_pairs,
                inputs=[
                    "modelling.feat.rtx_kg2",
                    "modelling.model_input.splits",
                    "params:matrix_generation.matrix_generation_options.drug_flags",
                    "params:matrix_generation.matrix_generation_options.disease_flags",
                ],
                outputs="matrix_generation.prm.pairs",
                name="generate_pairs",
            ),
        ]
    )
    pipes = [initial_node]
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_names = [model["model_name"] for model in models]
    for model in model_names:
        pipes.append(
            pipeline(
                _create_matrix_generation_pipeline(model),
                tags=model,
            )
        )

    return sum(pipes)
