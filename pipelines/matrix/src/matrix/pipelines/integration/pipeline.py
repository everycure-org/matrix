from kedro.pipeline import Pipeline, node, pipeline

from matrix import settings
from matrix.pipelines.batch import pipeline as batch_pipeline

from . import nodes


def _create_integration_pipeline(source: str, has_nodes: bool = True, has_edges: bool = True) -> Pipeline:
    pipelines = []

    pipelines.append(
        pipeline(
            [
                node(
                    func=nodes.transform,
                    inputs={
                        "transformer": f"params:integration.sources.{source}.transformer",
                        # NOTE: The datasets below are currently only picked up by RTX
                        # the goal is to ensure that semmed filtering occurs for all
                        # graphs in the future.
                        "curie_to_pmids": "ingestion.int.rtx_kg2.curie_to_pmids",
                        "semmed_filters": "params:integration.preprocessing.rtx.semmed_filters",
                        # NOTE: This dynamically wires the nodes and edges into each transformer.
                        # This is due to the fact that the Transformer objects are only created
                        # during node execution time, otherwise we could infer this based on
                        # the transformer.
                        **({"nodes_df": f"ingestion.int.{source}.nodes"} if has_nodes else {}),
                        **({"edges_df": f"ingestion.int.{source}.edges"} if has_edges else {}),
                    },
                    outputs={
                        "nodes": f"integration.int.{source}.nodes",
                        **({"edges": f"integration.int.{source}.edges"} if has_edges else {}),
                    },
                    name=f"transform_{source}_nodes",
                    tags=["standardize"],
                ),
                batch_pipeline.create_pipeline(
                    source=f"source_{source}",
                    df=f"integration.int.{source}.nodes",
                    output=f"integration.int.{source}.nodes.nodes_norm_mapping",
                    bucket_size="params:integration.normalization.batch_size",
                    transformer="params:integration.normalization.normalizer",
                    max_workers=120,
                ),
                node(
                    func=nodes.normalize_nodes,
                    inputs={
                        "mapping_df": f"integration.int.{source}.nodes.nodes_norm_mapping",
                        "nodes": f"integration.int.{source}.nodes",
                    },
                    outputs=f"integration.int.{source}.nodes.norm@spark",
                    name=f"normalize_{source}_nodes",
                ),
            ],
            tags=source,
        )
    )

    if has_edges:
        pipelines.append(
            pipeline(
                [
                    node(
                        func=nodes.normalize_edges,
                        inputs={
                            "mapping_df": f"integration.int.{source}.nodes.nodes_norm_mapping",
                            "edges": f"integration.int.{source}.edges",
                        },
                        outputs=f"integration.int.{source}.edges.norm@spark",
                        name=f"normalize_{source}_edges",
                    ),
                ],
                tags=source,
            )
        )

    return sum(pipelines)


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""

    pipelines = []

    # Create pipeline per source
    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration"):
        pipelines.append(
            pipeline(
                _create_integration_pipeline(
                    source=source["name"],
                    has_nodes=source.get("has_nodes", True),
                    has_edges=source.get("has_edges", True),
                ),
                tags=[source["name"]],
            )
        )

    # Add integration pipeline
    pipelines.append(
        pipeline(
            [
                node(
                    func=nodes.union_and_deduplicate_nodes,
                    inputs=[
                        "params:integration.deduplication.retrieve_most_specific_category",
                        *[
                            f'integration.int.{source["name"]}.nodes.norm@spark'
                            for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration")
                            if source.get("integrate_in_kg", True)
                        ],
                    ],
                    outputs="integration.prm.unified_nodes",
                    name="create_prm_unified_nodes",
                ),
                # union edges
                node(
                    func=nodes.union_and_deduplicate_edges,
                    inputs=[
                        f'integration.int.{source["name"]}.edges.norm@spark'
                        for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration")
                        if source.get("integrate_in_kg", True)
                    ],
                    outputs="integration.prm.unified_edges",
                    name="create_prm_unified_edges",
                ),
                # filter nodes given a set of filter stages
                node(
                    func=nodes.prefilter_unified_kg_nodes,
                    inputs=[
                        "integration.prm.unified_nodes",
                        "params:integration.filtering.node_filters",
                    ],
                    outputs="integration.prm.prefiltered_nodes",
                    name="prefilter_prm_knowledge_graph_nodes",
                    tags=["filtering"],
                ),
                # filter edges given a set of filter stages
                node(
                    func=nodes.filter_unified_kg_edges,
                    inputs=[
                        "integration.prm.prefiltered_nodes",
                        "integration.prm.unified_edges",
                        "params:integration.filtering.edge_filters",
                    ],
                    outputs="integration.prm.filtered_edges",
                    name="filter_prm_knowledge_graph_edges",
                    tags=["filtering"],
                ),
                node(
                    func=nodes.filter_nodes_without_edges,
                    inputs=[
                        "integration.prm.prefiltered_nodes",
                        "integration.prm.filtered_edges",
                    ],
                    outputs="integration.prm.filtered_nodes",
                    name="filter_nodes_without_edges",
                    tags=["filtering"],
                ),
                node(
                    func=nodes.add_gt_category,
                    inputs=[
                        "integration.prm.filtered_nodes",
                        "integration.int.drug_list.nodes.norm@spark",
                        "integration.int.disease_list.nodes.norm@spark",
                        *[
                            f"integration.int.{source['name']}.edges.norm@spark"
                            for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration")
                            if not source.get("integrate_in_kg", True) and not source.get("nodes_only", False)
                        ],
                    ],
                    outputs="integration.prm.filtered_gt_nodes",
                    name="add_gt_category",
                    tags=["tag-gt"],
                ),
            ]
        )
    )

    return sum(pipelines)
