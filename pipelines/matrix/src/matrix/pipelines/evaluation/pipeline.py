"""Evaluation pipeline."""
import pandas as pd

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
                    "modelling.feat.rtx_kg2",
                    "modelling.model_input.splits",
                    f"params:evaluation.{evaluation}.evaluation_options.generator",
                ],
                outputs=f"evaluation.{model}.{evaluation}.prm.pairs",
                name=f"create_{model}_{evaluation}_evaluation_pairs",
            ),
            node(
                func=nodes.make_test_predictions,
                inputs=[
                    "modelling.feat.rtx_kg2",
                    f"evaluation.{model}.{evaluation}.prm.pairs",
                    f"modelling.{model}.model_input.transformers",
                    f"modelling.{model}.models.model",
                    f"params:modelling.{model}.model_options.model_tuning_args.features",
                    f"params:evaluation.score_col_name",
                ],
                outputs=f"evaluation.{model}.{evaluation}.model_output.predictions",
                name=f"create_{model}_{evaluation}_model_predictions",
            ),
            node(
                func=nodes.evaluate_test_predictions,
                inputs=[
                    f"evaluation.{model}.{evaluation}.model_output.predictions",
                    f"params:evaluation.{evaluation}.evaluation_options.evaluation",
                ],
                outputs=f"evaluation.{model}.{evaluation}.reporting.evaluation",
                name=f"create_{model}_{evaluation}_evaluation",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create fabricator pipeline."""
    pipes = []
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_names = [model["model_name"] for model in models]
    for model in model_names:
        for evaluation in settings.DYNAMIC_PIPELINES_MAPPING.get("evaluation"):
            pipes.append(
                pipeline(
                    _create_evaluation_pipeline(model, evaluation["evaluation_name"]),
                    tags=model,
                )
            )
    # Consolidate reports
    pipes.append(
        pipeline(
            [
                node(
                    func=nodes.consolidate_evaluation_reports,
                    inputs=[
                        f"evaluation.{model['model_name']}.{evaluation['evaluation_name']}.reporting.evaluation"
                        for model in settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
                        for evaluation in settings.DYNAMIC_PIPELINES_MAPPING.get(
                            "evaluation"
                        )
                    ],
                    outputs=f"evaluation.reporting.report",
                    name=f"consolidate_reports",
                    tags=model_names,
                ),
            ]
        )
    )

    return sum(pipes)
