from matrix import settings
from kedro.pipeline import Pipeline, pipeline

from matrix.kedro4argo_node import argo_node
from . import nodes as nd
from ..matrix_generation.pipeline import create_pipeline as matrix_generation_pipeline


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
            argo_node(
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


def _create_inference_pipeline(model_excl: str, model_incl: str) -> Pipeline:
    """Matrix generation pipeline adjusted for running inference with models of choice."""
    mg_pipeline = matrix_generation_pipeline()

    # Unpack number of splits
    n_cross_val_folds = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation").get("n_cross_val_folds")

    inference_nodes = pipeline(
        [
            node
            for node in mg_pipeline.nodes
            if (
                not any(model in node.name for model in model_excl)  # Remove nodes for models not included in inference
                and not any(
                    f"fold_{fold}" in node.name for fold in range(n_cross_val_folds)
                )  # Include only models trained on full ground truth data
            )
        ]
    )
    pipelines = []
    for model in model_incl:
        pipelines.append(
            pipeline(
                [inference_nodes],
                parameters={
                    "params:evaluation.treat_score_col_name": "params:inference.score_col_name",
                    "params:matrix_generation.matrix_generation_options.batch_by": "params:inference.matrix_generation_options.batch_by",
                    "params:matrix_generation.matrix_generation_options.n_reporting": "params:inference.matrix_generation_options.n_reporting",
                },
                inputs={
                    "integration.int.drug_list.nodes@spark": "inference.int.drug_list@spark",
                    "integration.int.disease_list.nodes@spark": "inference.int.disease_list@spark",
                    "integration.int.drug_list.nodes@pandas": "inference.int.drug_list@pandas",
                    "integration.int.disease_list.nodes@pandas": "inference.int.disease_list@pandas",
                },
                outputs={
                    f"matrix_generation.{model}.fold_{n_cross_val_folds}.model_output.sorted_matrix_predictions@pandas": f"inference.{model}.model_output.predictions@pandas",
                    f"matrix_generation.{model}.fold_{n_cross_val_folds}.reporting.matrix_report": f"inference.{model}.reporting.report",
                    f"matrix_generation.prm.fold_{n_cross_val_folds}.matrix_pairs": "inference.prm.matrix_pairs",
                    "matrix_generation.feat.nodes_kg_ds": "inference.feat.nodes_kg_ds",
                    "matrix_generation.feat.nodes@spark": "inference.feat.nodes@spark",
                },
            )
        )
    return sum([*pipelines])


def _create_reporting_pipeline(model: str) -> Pipeline:
    """Reporting nodes of the inference pipeline for visualisation purposes."""
    return pipeline(
        [
            argo_node(
                func=nd.visualise_treat_scores,
                inputs={
                    "scores": f"inference.{model}.model_output.predictions@pandas",
                    "infer_type": "inference.int.request_type",
                    "col_name": "params:inference.score_col_name",
                },
                outputs=f"inference.{model}.reporting.visualisations",
                name=f"visualise_inference_{model}",
            )
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create requests pipeline.

    The pipelines is composed of static_nodes (i.e. nodes which are run only once at the beginning),
    and dynamic nodes (i.e. nodes which are repeated for each model selected).
    """
    # Get models of interest for inference
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_names_excl = [model for model, config in models.items() if not config["run_inference"]]
    model_names_incl = [model for model, config in models.items() if config["run_inference"]]

    # Construct the full pipeline
    resolution_nodes = _create_resolution_pipeline()
    inference_nodes = _create_inference_pipeline(model_names_excl, model_names_incl)
    pipelines = [resolution_nodes, inference_nodes]

    # Add reporting nodes for each model
    for model in model_names_incl:
        pipelines.append(
            pipeline(
                _create_reporting_pipeline(model),
                tags=model,
            )
        )

    return sum([*pipelines])
