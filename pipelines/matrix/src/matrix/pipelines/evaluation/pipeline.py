from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import argo_node

from . import nodes


def _create_evaluation_pipeline(model: str, evaluation: str) -> Pipeline:
    return pipeline(
        [
            argo_node(
                func=nodes.generate_test_dataset,
                inputs=[
                    f"matrix_generation.{model}.model_output.sorted_matrix_predictions@pandas",
                    f"params:evaluation.{evaluation}.evaluation_options.generator",
                ],
                outputs=f"evaluation.{model}.{evaluation}.model_output.pairs",
                name=f"create_{model}_{evaluation}_evaluation_pairs",
            ),
            argo_node(
                func=nodes.evaluate_test_predictions,
                inputs=[
                    f"evaluation.{model}.{evaluation}.model_output.pairs",
                    f"params:evaluation.{evaluation}.evaluation_options.evaluation",
                ],
                outputs=f"evaluation.{model}.{evaluation}.model_output.result",
                name=f"create_{model}_{evaluation}_evaluation",
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{model}.{evaluation}"],
    )


def _create_stability_pipeline(model_1: str, model_2: str, evaluation: str) -> Pipeline:
    return pipeline(
        [
            argo_node(
                func=nodes.generate_overlapping_dataset,
                inputs=[
                    f"matrix_generation.{model_1}.model_output.sorted_matrix_predictions@pandas",
                    f"matrix_generation.{model_2}.model_output.sorted_matrix_predictions@pandas",
                    f"params:evaluation.{evaluation}.evaluation_options.stability",
                ],
                outputs=f"evaluation.{model_1}.{model_2}.{evaluation}.model_output.pairs",
                name=f"create_{model_1}_{model_2}_{evaluation}_evaluation_pairs",
            ),
            argo_node(
                func=nodes.evaluate_stability_predictions,
                inputs=[
                    f"matrix_generation.{model_1}.model_output.sorted_matrix_predictions@pandas",
                    f"matrix_generation.{model_2}.model_output.sorted_matrix_predictions@pandas",
                    f"params:evaluation.{evaluation}.evaluation_options.stability",
                ],
                outputs=f"evaluation.{model_1}.{model_2}.{evaluation}.model_output.result",
                name=f"calculate_{model_1}_{model_2}_{evaluation}",
            ),
        ],
        tags=["stability-metrics"],
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create evaluation pipeline."""
    pipelines = []
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_names = [model["model_name"] for model in models]
    for model in model_names:
        pipelines.append(
            pipeline(
                [
                    argo_node(
                        func=nodes.perform_matrix_checks,
                        inputs=[
                            f"matrix_generation.{model}.model_output.sorted_matrix_predictions@pandas",
                            "modelling.model_input.splits",
                            "params:evaluation.score_col_name",
                        ],
                        outputs=None,
                        name=f"perform_{model}_matrix_checks",
                        tags="matrix_checks",
                    )
                ]
            )
        )
        for evaluation in settings.DYNAMIC_PIPELINES_MAPPING.get("evaluation"):
            pipelines.append(
                pipeline(
                    _create_evaluation_pipeline(model, evaluation["evaluation_name"]),
                    tags=model,
                )
            )
        for stability in settings.DYNAMIC_PIPELINES_MAPPING.get("stability"):
            for model_to_compare in model_names:
                pipelines.append(
                    pipeline(
                        _create_stability_pipeline(model, model_to_compare, stability["stability_name"]),
                        tags=model,
                    )
                )
    return sum(pipelines)
