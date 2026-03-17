from argo_kedro.pipeline import FusedPipeline, Node, sum_pipelines
from argo_kedro.pipeline.node import Node
from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.pipelines.batch import pipeline as batch_pipeline

from . import connectivity_metrics, nodes


def _create_integration_pipeline(
    source: str,
    has_nodes: bool = True,
    has_edges: bool = True,
    is_core: bool = False,
) -> Pipeline:
    return FusedPipeline(
        [
            Node(
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
                    **(
                        {
                            "positive_edges_df": f"ingestion.int.{source}.positive.edges@spark",
                            "negative_edges_df": f"ingestion.int.{source}.negative.edges@spark",
                        }
                        if ("ground_truth" in source)
                        else {"edges_df": f"ingestion.int.{source}.edges"}
                        if has_edges
                        else {}
                    ),
                },
                outputs={
                    "nodes": f"integration.int.{source}.nodes",
                    **({"edges": f"integration.int.{source}.edges"} if has_edges else {}),
                },
                name=f"transform_{source}_nodes",
                tags=["standardize"],
            ),
            batch_pipeline.cached_api_enrichment_pipeline(
                source=f"normalization_source_{source}",
                workers=20,
                input=f"integration.int.{source}.nodes",
                output=f"integration.int.{source}.nodes.nodes_norm_mapping",
                cache_miss_resolver="params:integration.normalization.normalizer",
                new_col="params:integration.normalization.target_col",
                preprocessor="params:integration.normalization.preprocessor",
                primary_key="params:integration.normalization.primary_key",
                batch_size="params:integration.normalization.batch_size",
                cache_schema="params:integration.normalization.cache_schema",
            ),
            Node(
                func=nodes.normalization_summary_nodes_and_edges
                if has_edges
                else nodes.normalization_summary_nodes_only,
                inputs={
                    "nodes": f"integration.int.{source}.nodes.norm@spark",
                    "mapping_df": f"integration.int.{source}.nodes.nodes_norm_mapping",
                    **({"edges": f"integration.int.{source}.edges.norm@spark"} if has_edges else {}),
                    "source": f"params:integration.sources.{source}.name",
                },
                outputs=f"integration.int.{source}.normalization_summary",
                name=f"create_{source}_normalization_summary",
                tags=["normalization"],
            ),
            *(
                [
                    Node(
                        func=nodes.normalize_edges,
                        inputs={
                            "mapping_df": f"integration.int.{source}.nodes.nodes_norm_mapping",
                            "edges": f"integration.int.{source}.edges",
                        },
                        outputs=f"integration.int.{source}.edges.norm@spark",
                        name=f"normalize_{source}_edges",
                    ),
                ]
                if has_edges
                else []
            ),
            Node(
                func=nodes.normalize_core_nodes if is_core else nodes.normalize_nodes,
                inputs={
                    "mapping_df": f"integration.int.{source}.nodes.nodes_norm_mapping",
                    "nodes": f"integration.int.{source}.nodes",
                },
                outputs=f"integration.int.{source}.nodes.norm@spark",
                name=f"normalize_{source}_nodes",
            ),
        ],
        name=f"integrate_{source}_fused",
        tags=source,
        machine_type="c4-highmem-16",
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""

    pipelines = []

    # Create pipeline per source
    for source in settings.DYNAMIC_PIPELINES_MAPPING().get("integration"):
        pipelines.append(
            pipeline(
                _create_integration_pipeline(
                    source=source["name"],
                    has_nodes=source.get("has_nodes", True),
                    has_edges=source.get("has_edges", True),
                    is_core=source.get("is_core", False),
                ),
                tags=[source["name"]],
            )
        )
    # Add integration pipeline
    pipelines.append(
        FusedPipeline(
            [
                Node(
                    func=nodes.create_core_id_mapping,
                    inputs=[
                        *[
                            f"integration.int.{source['name']}.nodes.norm@spark"
                            for source in settings.DYNAMIC_PIPELINES_MAPPING().get("integration")
                            if source.get("is_core", False)
                        ]
                    ],
                    outputs="integration.int.core_node_mapping",
                    name="create_core_id_mapping",
                ),
                Node(
                    func=nodes.union_and_deduplicate_nodes,
                    inputs=[
                        "params:integration.deduplication.retrieve_most_specific_category",
                        "integration.int.core_node_mapping",
                        *[
                            f"integration.int.{source['name']}.nodes.norm@spark"
                            for source in settings.DYNAMIC_PIPELINES_MAPPING().get("integration")
                            if source.get("integrate_in_kg", True)
                        ],
                    ],
                    outputs="integration.prm.unified_nodes",
                    name="create_prm_unified_nodes",
                ),
                Node(
                    func=nodes.unify_ground_truth,
                    inputs=[
                        *[
                            f"integration.int.{source['name']}.edges.norm@spark"
                            for source in settings.DYNAMIC_PIPELINES_MAPPING().get("integration")
                            if "ground_truth" in source["name"]
                        ],
                    ],
                    outputs="integration.int.unified_ground_truth_edges",
                    name="create_unified_ground_truth_edges",
                ),
                Node(
                    func=nodes.union_edges,
                    inputs=[
                        "integration.int.core_node_mapping",
                        *[
                            f"integration.int.{source['name']}.edges.norm@spark"
                            for source in settings.DYNAMIC_PIPELINES_MAPPING().get("integration")
                            if source.get("integrate_in_kg", True)
                        ],
                    ],
                    outputs="integration.prm.unified_edges",
                    name="create_prm_unified_edges",
                ),
                Node(
                    func=nodes._union_datasets,
                    inputs=[
                        *[
                            f"integration.int.{source['name']}.normalization_summary"
                            for source in settings.DYNAMIC_PIPELINES_MAPPING().get("integration")
                        ]
                    ],
                    outputs="integration.prm.unified_normalization_summary",
                    name="create_unified_normalization_summary",
                ),
                Node(
                    func=nodes.check_nodes_and_edges_matching,
                    inputs={
                        "nodes": "integration.prm.unified_nodes",
                        "edges": "integration.prm.unified_edges",
                    },
                    outputs="integration.prm.nodes_edges_consistency_check",
                    name="check_merged_nodes_and_edges_consistency",
                    tags=["validation"],
                ),
                Node(
                    func=nodes.compute_abox_tbox_metric,
                    inputs={
                        "edges": "integration.prm.unified_edges",
                    },
                    outputs="integration.prm.metric_abox_tbox",
                    name="metric_abox_tbox",
                    tags=["metrics", "ontological"],
                ),
                Node(
                    func=nodes.compute_ontology_inclusion_metric,
                    inputs={
                        "nodes": "integration.prm.unified_nodes",
                        "edges": "integration.prm.unified_edges",
                    },
                    outputs="integration.prm.node_ontology",
                    name="compute_ontology_inclusion_metric",
                    tags=["metrics", "ontological"],
                ),
                Node(
                    func=connectivity_metrics.compute_connected_components,
                    inputs={
                        "nodes": "integration.prm.unified_nodes",
                        "edges": "integration.prm.unified_edges",
                        "algorithm": "params:integration.connectivity.algorithm",
                        "core_id_mapping": "integration.int.core_node_mapping",
                    },
                    outputs={
                        "node_assignments": "integration.prm.node_components",
                        "component_stats": "integration.prm.connected_components_stats",
                    },
                    name="compute_connected_components",
                    tags=["metrics", "connectivity"],
                ),
                Node(
                    func=connectivity_metrics.compute_component_summary,
                    inputs="integration.prm.node_metrics",
                    outputs="integration.prm.connected_components",
                    name="compute_component_summary",
                    tags=["metrics", "connectivity"],
                ),
                Node(
                    func=connectivity_metrics.compute_core_connectivity_metrics,
                    inputs={
                        "nodes": "integration.prm.unified_nodes",
                        "core_id_mapping": "integration.int.core_node_mapping",
                        "connected_components": "integration.prm.node_metrics",
                    },
                    outputs="integration.prm.metric_core_connectivity_summary",
                    name="compute_core_connectivity_metrics",
                    tags=["metrics", "connectivity"],
                ),
                Node(
                    func=nodes.combine_node_metrics,
                    inputs={
                        "node_components": "integration.prm.node_components",
                        "node_ontology": "integration.prm.node_ontology",
                    },
                    outputs="integration.prm.node_metrics",
                    name="combine_node_metrics",
                    tags=["metrics"],
                ),
            ],
            name="integrate-sources-fused",
            machine_type="c4-highmem-16",
        )
    )
    return sum_pipelines(pipelines)
