from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import publish_dataset_to_hf, verify_published_dataset


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=publish_dataset_to_hf,
                inputs="integration.prm.unified_edges_pandas",
                outputs="kg_edges_hf_published",
                name="publish_kg_edges_node",
            ),

            node(
                func=verify_published_dataset,
                inputs="kg_edges_hf_published_read",
                outputs="publication_stats_edges",
                name="verify_publication_edges",
            ),

            node(
                func=publish_dataset_to_hf,
                inputs="integration.prm.unified_nodes_pandas",
                outputs="kg_nodes_hf_published",
                name="publish_kg_nodes_node",
            ),

            node(
                func=verify_published_dataset,
                inputs="kg_nodes_hf_published_read",
                outputs="publication_stats_nodes",
                name="verify_publication_nodes",
            ),
        ]
    )
