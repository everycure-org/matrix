from kedro.pipeline import Pipeline, node

from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    pipeline_nodes = [
        ArgoNode(
            func=nodes.prefilter_unified_kg_nodes,
            inputs=[
                "integration.prm.unified_nodes",
                "params:filtering.node_filters",
            ],
            outputs=["filtering.prm.prefiltered_nodes", "filtering.prm.removed_nodes_initial"],
            name="prefilter_prm_knowledge_graph_nodes",
            tags=[
                "argowf.fuse",
                "argowf.fuse-group.filtering",
            ],
            argo_config=ArgoResourceConfig(
                memory_limit=75,
                memory_request=75,
            ),
        ),
        node(
            func=nodes.filter_edges_without_nodes,
            inputs=[
                "filtering.prm.prefiltered_nodes",
                "integration.prm.unified_edges",
            ],
            outputs=["filtering.int.filtered_edges"],
            name="filter_edges_without_nodes",
            tags=[
                "argowf.fuse",
                "argowf.fuse-group.filtering",
            ],
        ),
        node(
            func=nodes.filter_kg_edges,
            inputs=[
                "filtering.int.filtered_edges",
                "params:filtering.edge_filters",
            ],
            outputs=["filtering.prm.filtered_edges", "filtering.prm.removed_edges"],
            name="filter_prm_knowledge_graph_edges",
            tags=[
                "argowf.fuse",
                "argowf.fuse-group.filtering",
            ],
        ),
        node(
            func=nodes.filter_nodes_without_edges,
            inputs=[
                "filtering.prm.prefiltered_nodes",
                "filtering.prm.filtered_edges",
            ],
            outputs=["filtering.prm.filtered_nodes", "filtering.prm.removed_nodes_final"],
            name="filter_nodes_without_edges",
            tags=[
                "argowf.fuse",
                "argowf.fuse-group.filtering",
            ],
        ),
    ]

    return Pipeline(pipeline_nodes)
