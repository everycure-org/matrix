import pyspark.sql as ps
from kedro.pipeline import Pipeline, pipeline
from matrix.kedro4argo_node import ArgoNode

from . import nodes
from .settings import RUN_COMPARISON_SETTINGS

matrices_to_evaluate = RUN_COMPARISON_SETTINGS["run_comparison"]["inputs"]
evaluations_to_run = RUN_COMPARISON_SETTINGS["run_comparison"]["evaluations"]


def _create_evaluation_pipeline(evaluation: str, matrix: ps.DataFrame) -> Pipeline:
    pipeline_nodes = [
        ArgoNode(
            func=nodes.run_evaluation,
            inputs=[
                matrix,
                f"params:run_comparison_evaluations.{evaluation}",
                f"params:run_comparison_evaluations.{evaluation}.bool_test_col",
                f"params:run_comparison_evaluations.{evaluation}.score_col",
            ],
            outputs=f"run_comparison.{matrix}.{evaluation}",
            name=f"cross_run_comparison.{matrix}.{evaluation}",
        )
    ]
    return pipeline_nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create cross-run comparison evaluation pipeline."""

    pipeline_nodes = []

    matrices = [x for x in matrices_to_evaluate.keys()]
    evaluations = [ev["evaluation_name"] for ev in evaluations_to_run]

    for matrix in matrices:
        for evaluation in evaluations:
            pipeline_nodes.append(pipeline(_create_evaluation_pipeline(evaluation, matrix)))
    return Pipeline(pipeline_nodes)
