from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import ArgoNode

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create matrix transformations pipeline."""

    # Is this the correct way to get the max fold?
    # Long term - sort out how to store fold data and full matrix data reproducibly regadless of number of folds
    max_fold = settings.DYNAMIC_PIPELINES_MAPPING().get("cross_validation")["n_cross_val_folds"]

    pipelines = [
        pipeline(
            [
                ArgoNode(
                    func=nodes.shuffle_and_limit_matrix,
                    inputs=[
                        f"matrix_generation.fold_{max_fold}.model_output.sorted_matrix_predictions@spark",
                    ],
                    outputs="matrix_transformations.shuffled_matrix@spark",
                    name="shuffle_and_limit_matrix",
                ),
                ArgoNode(
                    func=nodes.apply_matrix_transformations,
                    inputs={
                        "matrix": "matrix_transformations.shuffled_matrix@spark",
                        "transformations": "params:matrix_transformations.transformations",
                    },
                    outputs="matrix_transformations.full_matrix_transformed@spark",
                    name="apply_matrix_transformations",
                ),
            ]
        )
    ]

    return sum(pipelines)
