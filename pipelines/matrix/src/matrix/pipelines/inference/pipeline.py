from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import ArgoNode

from ..matrix_generation.pipeline import create_pipeline as matrix_generation_pipeline
from . import nodes as nd


def _create_resolution_pipeline() -> Pipeline:
    """Resolution pipeline for filtering out the input."""
    return pipeline(
        [
            # TODO: Remove implement this node /modification of it thats consistent with integration/batch
            # node(
            #     func=nodes.clean_input_sheet,
            #     inputs={
            #         "input_df": "preprocessing.raw.infer_sheet",
            #         "endpoint": "params:preprocessing.translator.normalizer",
            #         "conflate": "params:integration.nodenorm.conflate",
            #         "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
            #         "batch_size": "params:integration.nodenorm.batch_size",
            #         "parallelism": "params:integration.nodenorm.parallelism",
            #     },
            #     outputs="inference.raw.normalized_inputs",
            #     name="clean_input_sheet",
            #     tags=["inference-input"],
            # ),
            ArgoNode(
                func=nd.resolve_input_sheet,
                inputs={
                    "input_sheet": "inference.raw.normalized_inputs",
                    "drug_sheet": "inference.raw.drug_list",
                    "disease_sheet": "inference.raw.disease_list",
                },
                outputs=[
                    "inference.int.request_type",
                    "inference.int.drug_list@pandas",
                    "inference.int.disease_list@pandas",
                ],
                name="resolve_input_sheet",
            ),
        ]
    )


def _create_inference_pipeline() -> Pipeline:
    """Matrix generation pipeline adjusted for running inference with ALL models."""

    n_cross_val_folds = settings.DYNAMIC_PIPELINES_MAPPING().get("cross_validation").get("n_cross_val_folds")

    # Get ALL models for inference
    all_models = settings.DYNAMIC_PIPELINES_MAPPING().get("modelling", [])

    if not all_models:
        raise ValueError("No models configured for inference.")

    mg_pipeline = matrix_generation_pipeline()
    inference_nodes = pipeline(
        [  # Include only models trained on full ground truth data
            node for node in mg_pipeline.nodes if f"fold_{n_cross_val_folds}" in node.name
        ]
    )

    # Create outputs mapping for all models
    outputs_mapping = {}
    for model in all_models:
        model_name = model["model_name"]
        outputs_mapping.update(
            {
                f"matrix_generation.fold_{n_cross_val_folds}.{model_name}.model_output.sorted_matrix_predictions@pandas": f"inference.{model_name}.model_output.predictions@pandas",
                f"matrix_generation.{model_name}.reporting.plots": f"inference.{model_name}.reporting.plots",
                f"matrix_generation.{model_name}.reporting.tables": f"inference.{model_name}.reporting.tables",
            }
        )

    # Add shared outputs
    outputs_mapping.update(
        {
            f"matrix_generation.prm.fold_{n_cross_val_folds}.matrix_pairs": "inference.prm.matrix_pairs",
            "matrix_generation.feat.nodes@spark": "inference.feat.nodes@spark",
        }
    )

    return pipeline(
        [inference_nodes],
        parameters={
            "params:evaluation.treat_score_col_name": "params:inference.score_col_name",
            "params:matrix_generation.matrix_generation_options.batch_by": "params:inference.matrix_generation_options.batch_by",
            "params:matrix_generation.matrix_generation_options.n_reporting": "params:inference.matrix_generation_options.n_reporting",
        },
        inputs={
            "ingestion.int.drug_list@spark": "inference.int.drug_list@spark",
            "ingestion.int.disease_list@spark": "inference.int.disease_list@spark",
            "ingestion.int.drug_list@pandas": "inference.int.drug_list@pandas",
            "ingestion.int.disease_list@pandas": "inference.int.disease_list@pandas",
        },
        outputs=outputs_mapping,
    )


def _create_reporting_pipeline() -> Pipeline:
    """Reporting nodes of the inference pipeline for visualisation purposes."""

    # Create reporting nodes for ALL models
    all_models = settings.DYNAMIC_PIPELINES_MAPPING().get("modelling", [])
    reporting_nodes = []

    for model in all_models:
        model_name = model["model_name"]
        reporting_nodes.append(
            ArgoNode(
                func=nd.visualise_treat_scores,
                inputs={
                    "scores": f"inference.{model_name}.model_output.predictions@pandas",
                    "infer_type": "inference.int.request_type",
                    "col_name": "params:inference.score_col_name",
                },
                outputs=f"inference.{model_name}.reporting.visualisations",
                name=f"{model_name}_visualise_inference",
            )
        )

    return pipeline(reporting_nodes)


def create_pipeline(**kwargs) -> Pipeline:
    """Create requests pipeline.

    The pipelines is composed of static_nodes (i.e. nodes which are run only once at the beginning),
    and dynamic nodes (i.e. nodes which are repeated for each model selected).
    """
    pipelines = [_create_resolution_pipeline(), _create_inference_pipeline(), _create_reporting_pipeline()]

    return sum(pipelines)
