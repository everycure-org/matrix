from kedro.pipeline import Pipeline, pipeline

from matrix.kedro4argo_node import ArgoNode

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create filtering pipeline."""

    return pipeline(
        [
            ArgoNode(
                func=nodes.filter_kg,
                # TODO Figure out what the correct input is
                inputs={
                    "nodes": "integration.prm.filtered_nodes",
                    "edges": "integration.prm.filtered_edges",
                },
                outputs=[
                    "filtering.prm.rtxkg2_filtered_nodes",
                    "filtering.prm.rtxkg2_filtered_edges",
                    "filtering.prm.robokop_filtered_nodes",
                    "filtering.prm.robokop_filtered_edges",
                ],
                name="filter_kg",
            ),
        ]
    )
