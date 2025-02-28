from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""

    pipelines = []

    pipelines.append(
        pipeline(
            [
                # node(
                #     func=nodes.hierarchical_deduplication,
                #     inputs=[
                #         # unified and deduplicated nodes
                #         "integration.prm.unified_nodes",
                #     ],
                #     outputs="filtering.prm.dedpulicated_nodes",
                #     name="hierarchical_dedpulicate_kg_nodes",
                #     tags=["filtering"],
                # ),
                # Dedup edges too
                # Filter sources to RTX or ROBOKOP
                # Getting this to work specifically, then will generalise
                # node(
                #     #
                #     func=nodes.prefilter_unified_kg_nodes,
                #     inputs=[
                #         # "filtering.prm.dedpulicated_nodes",
                #         "integration.prm.unified_nodes",
                #         "params:filtering.filter_sources",
                #     ],
                #     outputs="filtering.prm.prefiltered_nodes_by_source",
                #     name="prefilter_prm_knowledge_graph_nodes_by_source",
                #     tags=["filtering"],
                # ),
                node(
                    #
                    func=nodes.prefilter_unified_kg_nodes,
                    inputs=[
                        # unified and deduplicated nodes
                        "integration.prm.unified_nodes",
                        "params:filtering.filtering.node_source_filters",
                    ],
                    outputs="filtering.prm.prefiltered_nodes",
                    name="prefilter_prm_knowledge_graph_nodes",
                    tags=["filtering"],
                ),
                # node(
                #     func=nodes.filter_unified_kg_edges,
                #     inputs=[
                #         "filtering.prm.prefiltered_nodes",
                #         "integration.prm.unified_edges",
                #         "params:filtering.filtering.edge_filters",
                #     ],
                #     outputs="filtering.prm.filtered_edges",
                #     name="filter_prm_knowledge_graph_edges",
                #     tags=["filtering"],
                # ),
                # node(
                #     func=nodes.filter_nodes_without_edges,
                #     inputs=[
                #         "filtering.prm.prefiltered_nodes",
                #         "filtering.prm.filtered_edges",
                #     ],
                #     outputs="filtering.prm.filtered_nodes",
                #     name="filter_nodes_without_edges",
                #     tags=["filtering"],
                # ),
            ]
        )
    )

    return sum(pipelines)
