from kedro.pipeline import Pipeline
from matrix import settings
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig

from . import nodes

RUN_COMPARISON_RESOURCE_CONFIG = ArgoResourceConfig(
    cpu_limit=24,
    cpu_request=24,
    memory_limit=256,
    memory_request=256,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create cross-run comparison evaluation pipeline."""

    pipeline_nodes = []
    run_comparison_settings = settings.DYNAMIC_PIPELINES_MAPPING().get("run_comparison")
    evaluations_to_perform = run_comparison_settings["evaluations"]

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
                    "params:run_comparison.input_data.apply_harmonization",
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
