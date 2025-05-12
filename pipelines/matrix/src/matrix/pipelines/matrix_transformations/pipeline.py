from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig
from matrix.pipelines.modelling.utils import partial_fold

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create matrix transformations pipeline."""

    # Is this the correct way to get the max fold?
    # Long term - sort out how to store fold data and full matrix data reproducibly regadless of number of folds
    max_fold = settings.DYNAMIC_PIPELINES_MAPPING().get("cross_validation")["n_cross_val_folds"]

    pipelines = [
        pipeline(
            # TEMP: this node is just for testing some files locally
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
                    func=nodes.frequent_flyer_transformation,
                    inputs=[
                        "matrix_transformations.shuffled_matrix@spark",
                    ],
                    outputs="matrix_transformations.full_matrix_transformed@spark",
                    name="frequent_flyer_transformation",
                ),
            ]
        )
    ]

    return sum(pipelines)
