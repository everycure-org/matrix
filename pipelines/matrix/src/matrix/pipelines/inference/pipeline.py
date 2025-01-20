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
    """Matrix generation pipeline adjusted for running inference with models of choice."""

    n_cross_val_folds = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation").get("n_cross_val_folds")

    mg_pipeline = matrix_generation_pipeline()
    inference_nodes = pipeline(
        [  # Include only models trained on full ground truth data
            node for node in mg_pipeline.nodes if f"fold_{n_cross_val_folds}" in node.name
        ]
    )

    pipelines = []
    pipelines.append(
        pipeline(
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
            outputs={
                f"matrix_generation.fold_{n_cross_val_folds}.model_output.sorted_matrix_predictions@pandas": f"inference.model_output.predictions@pandas",
                f"matrix_generation.fold_{n_cross_val_folds}.reporting.matrix_report": f"inference.reporting.report",
                f"matrix_generation.prm.fold_{n_cross_val_folds}.matrix_pairs": "inference.prm.matrix_pairs",
                "matrix_generation.feat.nodes_kg_ds": "inference.feat.nodes_kg_ds",
                "matrix_generation.feat.nodes@spark": "inference.feat.nodes@spark",
            },
        )
    )
    return sum([*pipelines])


def _create_reporting_pipeline() -> Pipeline:
    """Reporting nodes of the inference pipeline for visualisation purposes."""
    return pipeline(
        [
            ArgoNode(
                func=nd.visualise_treat_scores,
                inputs={
                    "scores": "inference.model_output.predictions@pandas",
                    "infer_type": "inference.int.request_type",
                    "col_name": "params:inference.score_col_name",
                },
                outputs="inference.reporting.visualisations",
                name="visualise_inference",
            )
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create requests pipeline.

    The pipelines is composed of static_nodes (i.e. nodes which are run only once at the beginning),
    and dynamic nodes (i.e. nodes which are repeated for each model selected).
    """

    # Construct the full pipeline
    resolution_nodes = _create_resolution_pipeline()
    pipelines = [resolution_nodes]
    inference_nodes = _create_inference_pipeline()
    pipelines.append(inference_nodes)

    # Add reporting nodes for each model
    pipelines.append(
        pipeline(
            _create_reporting_pipeline(),
        )
    )

    return sum([*pipelines])
