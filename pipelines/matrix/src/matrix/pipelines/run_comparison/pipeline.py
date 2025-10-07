from typing import List

import pyspark.sql as ps
from kedro.pipeline import Pipeline, pipeline
from matrix.kedro4argo_node import ArgoNode

from . import nodes
from .settings import RUN_COMPARISON_SETTINGS

matrices_to_evaluate = RUN_COMPARISON_SETTINGS["run_comparison"]["inputs"]
evaluations_to_run = RUN_COMPARISON_SETTINGS["run_comparison"]["evaluations"]


def _create_evaluation_pipeline(evaluation: str, matrices: List[ps.DataFrame]) -> Pipeline:
    pipeline_nodes = [
        ArgoNode(
            func=nodes.run_evaluation,
            inputs=[
                f"params:run_comparison_evaluations.{evaluation}",
                *matrices,
            ],
            outputs=f"cross_run_comparison.{evaluation}.result",
            name=f"{evaluation}.create_{evaluation}_evaluation",
        )
    ]
    return pipeline_nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create cross-run comparison evaluation pipeline.

    The pipeline mirrors the evaluation pipeline's parameter-driven pattern.
    Evaluations to run are defined under `run_comparison` parameters in the
    catalog, and a single dispatcher node invokes the requested evaluations.
    """

    pipeline_nodes = []

    matrices = [x for x in matrices_to_evaluate.keys()]
    evaluations = [ev["evaluation_name"] for ev in evaluations_to_run]

    for evaluation in evaluations:
        pipeline_nodes.append(pipeline(_create_evaluation_pipeline(evaluation, matrices)))
    return Pipeline(pipeline_nodes)
