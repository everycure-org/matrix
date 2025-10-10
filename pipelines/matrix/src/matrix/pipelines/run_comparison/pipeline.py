from kedro.pipeline import Pipeline
from matrix.kedro4argo_node import ArgoNode

from . import nodes
from .settings import RUN_COMPARISON_SETTINGS

# TODO:
# - Add unit test for recall@n class
# - Add matrix harmonisation + unit tests
# - Test on real data
# - Add unit test for input path classes
# - Modify multi matrices dataset to copy matrices to pipeline data folder
# - Clean up PR


def create_pipeline(**kwargs) -> Pipeline:
    """Create cross-run comparison evaluation pipeline."""

    pipeline_nodes = []

    pipeline_nodes.append(
        ArgoNode(
            func=nodes.create_input_matrices_dataset,
            inputs=[
                "params:run_comparison.input_paths",
            ],
            outputs="run_comparison.input_matrices",
            name=f"create_input_matrices_dataset",
        ),
        ArgoNode(
            func=nodes.harmonize_matrices,
            inputs=[
                "run_comparison.input_matrices",
                "params:run_comparison.available_ground_truth_cols",
                "params:run_comparison.perform_multifold_uncertainty_estimation",
                "params:run_comparison.assert_data_consistency",
            ],
            outputs=["run_comparison.harmonized_matrices", "run_comparison.predictions_info"],
            name=f"harmonize_matrices",
        ),
    )
    for evaluation in RUN_COMPARISON_SETTINGS["evaluations"]:
        pipeline_nodes.extend(
            [
                ArgoNode(
                    func=nodes.run_evaluation,
                    inputs=[
                        "params:run_comparison.perform_multifold_uncertainty_estimation",
                        "params:run_comparison.perform_bootstrap_uncertainty_estimation",
                        f"params:run_comparison.evaluations.{evaluation}",
                        "run_comparison.harmonized_matrices",
                        "run_comparison.predictions_info",
                        "params:run_comparison.available_ground_truth_cols",
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
                        "run_comparison.harmonized_matrices",
                    ],
                    outputs=f"run_comparison.{evaluation}.plot",
                    name=f"plot_results.{evaluation}",
                ),
            ]
        )
    return Pipeline(pipeline_nodes)
