from argo_kedro.pipeline import FusedPipeline, Node
from kedro.pipeline import Pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create filtering pipeline."""
    return FusedPipeline(
        [
            Node(
                func=nodes.prefilter_unified_kg_nodes,
                inputs=[
                    "integration.prm.unified_nodes",
                    "params:filtering.node_filters",
                ],
                outputs=["filtering.prm.prefiltered_nodes", "filtering.prm.removed_nodes_initial"],
                name="prefilter_prm_knowledge_graph_nodes",
            ),
            Node(
                func=nodes.filter_unified_kg_edges,
                inputs=[
                    "filtering.prm.prefiltered_nodes",
                    "integration.prm.unified_edges",
                    "params:filtering.edge_filters",
                ],
                outputs=["filtering.prm.filtered_edges", "filtering.prm.removed_edges"],
                name="filter_prm_knowledge_graph_edges",
            ),
            Node(
                func=nodes.filter_nodes_without_edges,
                inputs=[
                    "filtering.prm.prefiltered_nodes",
                    "filtering.prm.filtered_edges",
                ],
                outputs=["filtering.prm.filtered_nodes", "filtering.prm.removed_nodes_final"],
                name="filter_nodes_without_edges",
            ),
        ],
        name="filtering_fused",
        tags=["filtering"],
        machine_type="c4-highmem-16",
    )
