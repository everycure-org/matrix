from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create matrix transformations pipeline."""

    n_cross_val_folds = settings.DYNAMIC_PIPELINES_MAPPING().get("cross_validation")["n_cross_val_folds"]

    pipelines = []
    for fold in range(n_cross_val_folds + 1):  # NOTE: final fold is full training data
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.apply_matrix_transformations,
                        inputs={
                            "matrix": f"matrix_generation.fold_{fold}.model_output.sorted_matrix_predictions@spark",
                            "transformations": "params:matrix_transformations.transformations",
                            "score_col": "params:matrix_transformations.score_col",
                        },
                        outputs=f"matrix_transformations.fold_{fold}.model_output.sorted_matrix_predictions@spark",
                        name=f"apply_matrix_transformations_fold_{fold}",
                        argo_config=ArgoResourceConfig(cpu_request=8, memory_request=64, memory_limit=64),
                    ),
                ]
            )
        )

    # Persist the final fold predictions (trained on complete dataset) for BigQuery export
    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.return_predictions,
                    inputs=[
                        f"matrix_transformations.fold_{n_cross_val_folds}.model_output.sorted_matrix_predictions@spark",
                    ],
                    outputs=f"matrix_transformations.full_matrix_output@spark",
                    name="store_transformed_predictions",
                ),
            ]
        )
    )

    return sum(pipelines)
