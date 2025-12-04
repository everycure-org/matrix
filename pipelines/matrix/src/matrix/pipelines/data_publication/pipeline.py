from __future__ import annotations

from kedro.pipeline import Pipeline, node


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=lambda x: x,
                inputs="integration.prm.unified_edges@pandas",
                outputs="kg_edges_hf_published",
                name="publish_kg_edges_node",
            ),
            node(
                func=lambda x: x,
                inputs="integration.prm.unified_nodes@pandas",
                outputs="kg_nodes_hf_published",
                name="publish_kg_nodes_node",
            ),
        ]
    )
