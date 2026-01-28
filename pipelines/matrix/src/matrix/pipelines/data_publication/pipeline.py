from __future__ import annotations

from kedro.pipeline import Pipeline, node


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=lambda x: x,
                inputs="integration.prm.unified_edges",
                outputs="data_publication.prm.kg_edges_hf_published",
                name="publish_kg_edges_node",
            ),
            node(
                func=lambda x: x,
                inputs="integration.prm.unified_nodes",
                outputs="data_publication.prm.kg_nodes_hf_published",
                name="publish_kg_nodes_node",
            ),
        ]
    )
