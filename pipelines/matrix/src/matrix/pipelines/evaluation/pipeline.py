"""Evaluation pipeline."""
from typing import List
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from matrix import settings

from . import nodes


def _create_evaluation_pipeline(model: str, evaluation: str) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.generate_test_dataset,
                inputs=[
                    f"matrix_generation.{model}.model_output.sorted_matrix_predictions",
                    f"params:evaluation.{evaluation}.evaluation_options.generator",
                ],
                outputs=f"evaluation.{model}.{evaluation}.model_output.pairs",
                name=f"create_{model}_{evaluation}_evaluation_pairs",
            ),
            node(
                func=nodes.evaluate_test_predictions,
                inputs=[
                    f"evaluation.{model}.{evaluation}.model_output.pairs",
                    f"params:evaluation.{evaluation}.evaluation_options.evaluation",
                ],
                outputs=f"evaluation.{model}.{evaluation}.reporting.evaluation",
                name=f"create_{model}_{evaluation}_evaluation",
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{model}.{evaluation}"],
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create evaluation pipeline."""
    pipes = []
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_names = [model["model_name"] for model in models]
    for model in model_names:
        pipes.append(
            pipeline(
                [
                    node(
                        func=nodes.perform_matrix_checks,
                        inputs=[
                            f"matrix_generation.{model}.model_output.sorted_matrix_predictions",
                            "modelling.model_input.splits",
                            "params:evaluation.score_col_name",
                        ],
                        outputs=None,
                        tags="matrix_checks",
                    )
                ]
            )
        )
        for evaluation in settings.DYNAMIC_PIPELINES_MAPPING.get("evaluation"):
            pipes.append(
                pipeline(
                    _create_evaluation_pipeline(model, evaluation["evaluation_name"]),
                    tags=model,
                )
            )

    return sum(pipes)
