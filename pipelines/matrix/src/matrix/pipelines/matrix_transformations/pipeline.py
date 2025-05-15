from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import ArgoNode

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
                        },
                        outputs=f"matrix_transformations.fold_{fold}.transformed_matrix@spark",
                        name=f"apply_matrix_transformations_fold_{fold}",
                    ),
                ]
            )
        )

    return sum(pipelines)
