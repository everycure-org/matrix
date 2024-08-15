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
    for model in settings.DYNAMIC_PIPELINES_MAPPING.get("modelling"):
        for evaluation in settings.DYNAMIC_PIPELINES_MAPPING.get("evaluation"):
            pipes.append(
                pipeline(
                    _create_evaluation_pipeline(
                        model["model_name"], evaluation["evaluation_name"]
                    ),
                    tags=model["model_name"],
                )
            )
    # Consolidate reports
    pipes.append(
        pipeline(
            [
                # Example using new Generator
                node(
                    func=nodes.generator_example,
                    inputs=["params:evaluation.generator_with_dataset"],
                    outputs=None,
                    name="generator_example",
                ),
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
                ),
            ]
        )
    )

    return sum(pipes)


def _implement_time_split_validation(model_name: str) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.generate_time_split_validation_barplot,
                inputs=[
                    "modelling.feat.rtx_kg2",  # TODO: make it more flexible to fit any KGs
                    f"modelling.{model_name}.models.model",
                    model_name,
                    f"modelling.{model_name}.model_input.transformers",
                    "evaluation.cleaned_clinical_trial_data",
                ],
                outputs=f"evaluation.{model_name}.reporting.time_split_validation_barplot",
                name=f"generate_time_split_validation_barplot_{model_name}",
            ),
            node(
                func=nodes.generate_time_split_validation_classification_auroc,
                inputs=[
                    "modelling.feat.rtx_kg2",  # TODO: make it more flexible to fit any KGs
                    f"modelling.{model_name}.models.model",
                    model_name,
                    f"modelling.{model_name}.model_input.transformers",
                    "evaluation.cleaned_clinical_trial_data",
                ],
                outputs=f"evaluation.{model_name}.reporting.time_split_validation_classification_auroc",
                name=f"generate_time_split_validation_classification_auroc_{model_name}",
            ),
            node(
                func=nodes.generate_time_split_validation_all_metrics,
                inputs=[
                    "modelling.feat.rtx_kg2",  # TODO: make it more flexible to fit any KGs
                    f"modelling.{model_name}.models.model",
                    model_name,
                    f"modelling.{model_name}.model_input.transformers",
                    "evaluation.cleaned_clinical_trial_data",
                    "",  # TODO: need to check with Alexei where to get this data
                    "params:evaluation.time_split_validation.k_list_for_hit_at_k",
                    "params:evaluation.time_split_validation.clinical_label",
                ],
                outputs=f"evaluation.{model_name}.reporting.time_split_validation_all_metrics",
                name=f"generate_time_split_validation_all_metrics_{model_name}",
            ),
        ]
    )


def implement_time_split_validation_pipeline(**kwargs) -> Pipeline:
    """Implement time-split validation pipeline."""
    pipes = []

    # Clean clinical trial data
    pipes.append(
        pipeline(
            node(
                func=nodes.clean_clinical_trial_data,
                inputs=[
                    "evaluation.raw.medical",
                    "params:evaluation.synonymizer.endpoint",
                ],
                outputs="evaluation.cleaned_clinical_trial_data",
                name="cleaned_clinical_trial_data",
            ),
        )
    )

    # Run time split validation pipeline for different models
    auroc_output_name_lst = []
    all_metrics_output_name_lst = []
    for model in settings.DYNAMIC_PIPELINES_MAPPING.get("modelling"):
        pipes.append(
            pipeline(
                _implement_time_split_validation(model["model_name"]),
                tags=[model["model_name"], "time-split-validation"],
            )
        )
        auroc_output_name_lst.append(
            f"evaluation.{model['model_name']}.reporting.time_split_validation_classification_auroc"
        )
        all_metrics_output_name_lst.append(
            f"evaluation.{model['model_name']}.reporting.time_split_validation_all_metrics"
        )

    # Consolidate reports
    pipes.append(
        pipeline(
            [
                node(
                    func=nodes.create_metrics_report,
                    inputs=[auroc_output_name_lst],
                    outputs="evaluation.reporting.time_split_validation_classification_auroc",
                    name="create_time_split_validation_classification_auroc_report",
                ),
                node(
                    func=nodes.create_metrics_report,
                    inputs=[all_metrics_output_name_lst],
                    outputs="evaluation.reporting.time_split_validation_all_metrics",
                    name="create_time_split_validation_all_metrics_report",
                ),
            ]
        )
    )

    return sum(pipes)
