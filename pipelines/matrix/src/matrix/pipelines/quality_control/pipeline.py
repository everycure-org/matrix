import logging

from kedro.pipeline import Pipeline, pipeline

from matrix.kedro4argo_node import ArgoNode

from . import nodes

logger = logging.getLogger(__name__)


def integration_quality_control_pipeline() -> Pipeline:
    return pipeline(
        [
            ArgoNode(
                func=nodes.count_filtered_nodes,
                inputs="integration.prm.filtered_nodes",
                outputs="reporting.integration.filtered_nodes_agg_count",
                name="count_filtered_nodes",
                tags=["integration_quality_control"],
            ),
            ArgoNode(
                func=nodes.count_filtered_edges,
                inputs=["integration.prm.filtered_nodes", "integration.prm.filtered_edges"],
                outputs="reporting.integration.filtered_edges_agg_count",
                name="count_filtered_edges",
                tags=["integration_quality_control"],
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            integration_quality_control_pipeline(),
        ]
    )
