from kedro.pipeline import Pipeline, node

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    pipeline_nodes = [
        node(
            func=nodes.prefilter_unified_kg_nodes,
            inputs=[
                "integration.prm.unified_nodes",
                "params:filtering.node_filters",
            ],
            outputs="filtering.prm.prefiltered_nodes",
            name="prefilter_prm_knowledge_graph_nodes",
            tags=["filtering"],
        ),
        node(
            func=nodes.filter_unified_kg_edges,
            inputs=[
                "filtering.prm.prefiltered_nodes",
                "integration.prm.unified_edges",
                "params:filtering.edge_filters",
            ],
            outputs="filtering.prm.filtered_edges",
            name="filter_prm_knowledge_graph_edges",
            tags=["filtering"],
        ),
        node(
            func=nodes.filter_nodes_without_edges,
            inputs=[
                "filtering.prm.prefiltered_nodes",
                "filtering.prm.filtered_edges",
            ],
            outputs="filtering.prm.filtered_nodes",
            name="filter_nodes_without_edges",
            tags=["filtering"],
        ),
    ]

    return Pipeline(pipeline_nodes)
