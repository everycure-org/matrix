from kedro.pipeline import Pipeline
from matrix.kedro4argo_node import ArgoNode

from . import nodes
from .settings import RUN_COMPARISON_SETTINGS

# TODO:
# - Add bootstrap method
# - Add plotting method
# - Add unit test for recall@n class
# - Add matrix harmonisation + unit tests
# - Test on real data
# - Add unit test for input path classes
# - Modify multi matrices dataset to copy matrices to pipeline data folder

# matrices_to_evaluate = RUN_COMPARISON_SETTINGS["run_comparison"]["inputs"]


# def _create_evaluation_pipeline(evaluation: str, matrix: ps.DataFrame) -> Pipeline:
#     pipeline_nodes = [
#         ArgoNode(
#             func=nodes.run_evaluation,
#             inputs=[
#                 matrix,
#                 f"params:run_comparison_evaluations.{evaluation}",
#             ],
#             outputs=f"run_comparison.{matrix}.{evaluation}",
#             name=f"cross_run_comparison.{matrix}.{evaluation}",
#         )
#     ]
#     return pipeline_nodes


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
        )
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
                        "run_comparison.input_matrices",
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
                        "run_comparison.input_matrices",
                    ],
                    outputs=f"run_comparison.{evaluation}.plot",
                    name=f"plot_results.{evaluation}",
                ),
            ]
        )
    return Pipeline(pipeline_nodes)
