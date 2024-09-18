"""Fabricator pipeline."""
from matrix import settings
from kedro.pipeline import Pipeline, node, pipeline
from . import nodes as nd
from ..matrix_generation import nodes as matrix_gen

# FUTURE: move to data science runbook
# How to run inference pipeline using models in remote MLFlow:
# 1. Specify WORKFLOW_ID in .env which corresponds to the run name in MLFlow you want to use for inference
# 1.1 You can find the run names in mlflow https://mlflow.platform.dev.everycure.org/
# 2. Execute inference pipeline in cloud by executing the following
#  > kedro run --env cloud
# 2.1 Note that you might get an error when running the ingest_disease list node as API reaches max num of rows to be written.
# If that's the case you can simply re-run from that node onwards


def _create_inference_pipeline(model: str) -> Pipeline:
    """Part of the inference pipeline which gets re-executed for each model selected."""
    return pipeline(
        [
            node(
                func=matrix_gen.make_predictions_and_sort,
                inputs=[
                    "modelling.feat.rtx_kg2",
                    "inference.model_input.drug_disease_pairs",
                    f"modelling.{model}.model_input.transformers",
                    f"modelling.{model}.models.model",
                    f"params:modelling.{model}.model_options.model_tuning_args.features",
                    "params:inference.score_col_name",
                    "params:inference.matrix_generation_options.batch_by",
                ],
                outputs=f"inference.{model}.model_output.predictions",
                name=f"request_{model}_predictions_and_sort",
            ),
            node(
                func=matrix_gen.generate_report,
                inputs=[
                    f"inference.{model}.model_output.predictions",
                    "params:inference.matrix_generation_options.n_reporting",
                    "inference.raw.drug_list",
                    "inference.raw.disease_list",
                    "modelling.model_input.splits",
                    "params:inference.score_col_name",
                ],
                outputs=f"inference.{model}.reporting.report",
                name=f"add_metadata_{model}",
            ),
            # FUTURE: add describe_scores node once we get input from the medical team
            # node(func=nd.describe_scores)
            node(
                func=nd.visualise_treat_scores,
                inputs={
                    "scores": f"inference.{model}.model_output.predictions",
                    "infer_type": "inference.int.request_type",
                    "col_name": "params:inference.score_col_name",
                },
                outputs=f"inference.{model}.reporting.visualisations",
                name=f"visualise_inference_{model}",
            ),
        ],
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create requests pipeline.

    The pipelines is composed of static_nodes (i.e. nodes which are run only once at the beginning),
    and dynamic nodes (i.e. nodes which are repeated for each model selected).
    """
    static_nodes = pipeline(
        [
            node(
                func=lambda x: x,
                inputs="ingestion.raw.drug_list",
                outputs="inference.raw.drug_list",
                name=f"ingest_drug_list",
            ),
            node(
                func=lambda x: x,
                inputs="ingestion.raw.disease_list",
                outputs="inference.raw.disease_list",
                name=f"ingest_disease_list",
            ),
            node(
                func=nd.resolve_input_sheet,
                inputs={
                    "input_sheet": "inference.raw.normalized_inputs",
                    "drug_sheet": "inference.raw.drug_list",
                    "disease_sheet": "inference.raw.disease_list",
                },
                outputs=[
                    "inference.int.request_type",
                    "inference.int.drug_list",
                    "inference.int.disease_list",
                ],
                name="resolve_input_sheet",
            ),
            node(
                func=matrix_gen.generate_pairs,
                inputs=[
                    "inference.int.drug_list",
                    "inference.int.disease_list",
                    "modelling.model_input.splits",
                ],
                outputs=f"inference.model_input.drug_disease_pairs",
                name="generate_pairs_per_request",
            ),
        ]
    )
    pipes = [static_nodes]
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_names = [model["model_name"] for model in models if model["run_inference"]]
    for model in model_names:
        pipes.append(
            pipeline(
                _create_inference_pipeline(model),
                tags=model,
            )
        )
    print(sum([*pipes]))
    return sum([*pipes])
