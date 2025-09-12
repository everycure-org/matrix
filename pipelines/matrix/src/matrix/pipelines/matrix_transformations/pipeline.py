from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create matrix transformations pipeline."""

    n_cross_val_folds = settings.DYNAMIC_PIPELINES_MAPPING().get("cross_validation")["n_cross_val_folds"]

    # Get ALL models for transformation
    all_models = settings.DYNAMIC_PIPELINES_MAPPING().get("modelling", [])

    if not all_models:
        raise ValueError("No models configured for matrix transformations.")

    pipelines = []

    # Create transformations for ALL models
    for model in all_models:
        model_name = model["model_name"]

        for fold in range(n_cross_val_folds + 1):  # NOTE: final fold is full training data
            pipelines.append(
                pipeline(
                    [
                        ArgoNode(
                            func=nodes.apply_matrix_transformations,
                            inputs={
                                "matrix": f"matrix_generation.fold_{fold}.{model_name}.model_output.sorted_matrix_predictions@spark",
                                "transformations": "params:matrix_transformations.transformations",
                                "score_col": "params:matrix_transformations.score_col",
                            },
                            outputs=f"matrix_transformations.fold_{fold}.{model_name}.model_output.sorted_matrix_predictions@spark",
                            name=f"{model_name}_apply_matrix_transformations_fold_{fold}",
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
                        inputs={
                            "sorted_matrix_df": f"matrix_transformations.fold_{n_cross_val_folds}.{model_name}.model_output.sorted_matrix_predictions@spark",
                            "known_pairs": "modelling.model_input.splits@spark",
                        },
                        outputs=f"matrix_transformations.{model_name}.full_matrix_output@spark",
                        name=f"{model_name}_store_transformed_predictions",
                    ),
                ]
            )
        )

    return sum(pipelines)
