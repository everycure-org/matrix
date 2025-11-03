from kedro.pipeline import Pipeline
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig

from . import nodes
from .settings import RUN_COMPARISON_SETTINGS

RUN_COMPARISON_RESOURCE_CONFIG = ArgoResourceConfig(
    cpu_limit=24,
    cpu_request=24,
    memory_limit=256,
    memory_request=256,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create cross-run comparison evaluation pipeline."""

    pipeline_nodes = []

    evaluations_to_perform = [ev["name"] for ev in RUN_COMPARISON_SETTINGS["evaluations"] if ev["is_activated"]]

    pipeline_nodes.extend(
        [
            ArgoNode(
                func=nodes.process_input_filepaths,
                inputs=[
                    "params:run_comparison.input_data.input_paths",
                ],
                outputs="run_comparison.input_matrices",
                name=f"process_input_filepaths",
            ),
            ArgoNode(
                func=nodes.combine_matrix_pairs,
                inputs=[
                    "run_comparison.input_matrices",
                    "params:run_comparison.available_ground_truth_cols",
                    "params:run_comparison.perform_multifold_uncertainty_estimation",
                    "params:run_comparison.assert_data_consistency",
                ],
                outputs=["run_comparison.combined_pairs", "run_comparison.predictions_info"],
                name=f"combine_matrix_pairs",
                argo_config=RUN_COMPARISON_RESOURCE_CONFIG,
            ),
            ArgoNode(
                func=nodes.restrict_predictions,
                inputs=[
                    "run_comparison.input_matrices",
                    "run_comparison.combined_pairs",
                    "run_comparison.predictions_info",
                ],
                outputs="run_comparison.combined_predictions",
                name=f"restrict_predictions",
                argo_config=RUN_COMPARISON_RESOURCE_CONFIG,
            ),
        ]
    )
    for evaluation in evaluations_to_perform:
        pipeline_nodes.extend(
            [
                ArgoNode(
                    func=nodes.run_evaluation,
                    inputs=[
                        "params:run_comparison.perform_multifold_uncertainty_estimation",
                        "params:run_comparison.perform_bootstrap_uncertainty_estimation",
                        f"params:run_comparison.evaluations.{evaluation}",
                        "run_comparison.combined_predictions",
                        "run_comparison.predictions_info",
                    ],
                    outputs=f"run_comparison.{evaluation}.results",
                    name=f"give_evaluation_results.{evaluation}",
                ),
                ArgoNode(
                    func=nodes.plot_results,
                    inputs=[
                        "params:run_comparison.perform_multifold_uncertainty_estimation",
                        "params:run_comparison.perform_bootstrap_uncertainty_estimation",
                        f"params:run_comparison.evaluations.{evaluation}",
                        f"run_comparison.{evaluation}.results",
                        "run_comparison.combined_pairs",
                        "run_comparison.predictions_info",
                    ],
                    outputs=f"run_comparison.{evaluation}.plot",
                    name=f"plot_results.{evaluation}",
                ),
            ]
        )
    return Pipeline(pipeline_nodes)
