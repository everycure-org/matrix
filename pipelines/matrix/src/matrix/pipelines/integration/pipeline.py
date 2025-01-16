import pyspark.sql as ps
from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.inject import inject_object
from matrix.kedro4argo_node import ArgoNode

from . import nodes


@inject_object()
def transform_nodes(transformer, nodes_df: ps.DataFrame, **kwargs):
    return transformer.transform_nodes(nodes_df=nodes_df, **kwargs)


@inject_object()
def transform_edges(transformer, edges_df: ps.DataFrame, **kwargs):
    return transformer.transform_edges(edges_df=edges_df, **kwargs)


def _create_integration_pipeline(source: str) -> Pipeline:
    return pipeline(
        [
            ArgoNode(
                func=transform_nodes,
                inputs={
                    "transformer": f"params:integration.sources.{source}.transformer",
                    "nodes_df": f"ingestion.int.{source}.nodes",
                    "biolink_categories_df": "integration.raw.biolink.categories",
                },
                outputs=f"integration.int.{source}.nodes",
                name=f"transform_{source}_nodes",
                tags=["standardize"],
            ),
            ArgoNode(
                func=transform_edges,
                inputs={
                    "transformer": f"params:integration.sources.{source}.transformer",
                    "edges_df": f"ingestion.int.{source}.edges",
                    # NOTE: The datasets below are currently only picked up by RTX
                    # the goal is to ensure that semmed filtering occurs for all
                    # graphs in the future.
                    "curie_to_pmids": "ingestion.int.rtx_kg2.curie_to_pmids",
                    "semmed_filters": "params:integration.preprocessing.rtx.semmed_filters",
                },
                outputs=f"integration.int.{source}.edges",
                name=f"transform_{source}_edges",
                tags=["standardize"],
            ),
            # FUTURE: Extract normalizer technique
            ArgoNode(
                func=nodes.normalize_kg,
                inputs={
                    "nodes": f"integration.int.{source}.nodes",
                    "edges": f"integration.int.{source}.edges",
                    "api_endpoint": "params:integration.nodenorm.api_endpoint",
                    "conflate": "params:integration.nodenorm.conflate",
                    "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
                    "batch_size": "params:integration.nodenorm.batch_size",
                    "parallelism": "params:integration.nodenorm.parallelism",
                },
                outputs=[
                    f"integration.int.{source}.nodes.norm",
                    f"integration.int.{source}.edges.norm",
                    f"integration.int.{source}.nodes_norm_mapping",
                ],
                name=f"normalize_{source}_kg",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""

    # Create pipeline per source
    pipelines = []
    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration"):
        pipelines.append(
            pipeline(
                _create_integration_pipeline(source=source["name"]),
                tags=[source["name"]],
            )
        )

    # Add integration pipeline
    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.union_and_deduplicate_nodes,
                    inputs=[
                        "integration.raw.biolink.categories",
                        *[
                            f'integration.int.{source["name"]}.nodes.norm'
                            for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration")
                        ],
                    ],
                    outputs="integration.prm.unified_nodes",
                    name="create_prm_unified_nodes",
                ),
                # union edges
                ArgoNode(
                    func=nodes.union_and_deduplicate_edges,
                    inputs=[
                        f'integration.int.{source["name"]}.edges.norm'
                        for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration")
                    ],
                    outputs="integration.prm.unified_edges",
                    name="create_prm_unified_edges",
                ),
                # filter nodes given a set of filter stages
                ArgoNode(
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
                ArgoNode(
                    func=nodes.filter_unified_kg_edges,
                    inputs=[
                        "integration.prm.prefiltered_nodes",
                        "integration.prm.unified_edges",
                        "integration.raw.biolink.predicates",
                        "params:integration.filtering.edge_filters",
                    ],
                    outputs="integration.prm.filtered_edges",
                    name="filter_prm_knowledge_graph_edges",
                    tags=["filtering"],
                ),
                ArgoNode(
                    func=nodes.filter_nodes_without_edges,
                    inputs=[
                        "integration.prm.prefiltered_nodes",
                        "integration.prm.filtered_edges",
                    ],
                    outputs="integration.prm.filtered_nodes",
                    name="filter_nodes_without_edges",
                    tags=["filtering"],
                ),
                ArgoNode(
                    func=nodes.incorporate_feedback,
                    inputs=[
                        "ingestion.int.feedback@spark",
                        "integration.prm.prefiltered_nodes",
                        "integration.prm.filtered_edges",
                    ],
                    outputs=["integration.prm.feedback_filtered_nodes", "integration.prm.feedback_filtered_edges"],
                    name="incorporate_feedback",
                    tags=["feedback"],
                ),
            ]
        )
    )

    return sum(pipelines)
