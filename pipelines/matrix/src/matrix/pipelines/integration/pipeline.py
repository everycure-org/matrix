from typing import List

import pyspark.sql as ps
from kedro.pipeline import Pipeline, node, pipeline

from matrix import settings
from matrix.pipelines.batch import pipeline as batch_pipeline

from . import nodes


def _create_integration_pipeline(source: str, nodes_only: bool = False) -> Pipeline:
    pipelines = []

    pipelines.append(
        pipeline(
            [
                node(
                    func=nodes.transform_nodes,
                    inputs={
                        "transformer": f"params:integration.sources.{source}.transformer",
                        "nodes_df": f"ingestion.int.{source}.nodes",
                        "biolink_categories_df": "integration.raw.biolink.categories",
                    },
                    outputs=f"integration.int.{source}.nodes",
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
            ]
        )
    )

    if not nodes_only:
        pipelines.append(
            pipeline(
                [
                    node(
                        func=nodes.transform_edges,
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
                    node(
                        func=nodes.normalize_edges,
                        inputs={
                            "mapping_df": f"integration.int.{source}.nodes.nodes_norm_mapping",
                            "edges": f"integration.int.{source}.edges",
                        },
                        outputs=f"integration.int.{source}.edges.norm@spark",
                        name=f"normalize_{source}_edges",
                    ),
                ]
            )
        )

    return sum(pipelines)


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""

    # Create pipeline per source
    pipelines = []
    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration"):
        pipelines.append(
            pipeline(
                _create_integration_pipeline(source=source["name"], nodes_only=source.get("nodes_only", False)),
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
                            if source.get("integrate_in_kg", True) and not source.get("nodes_only", False)
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
                        if source.get("integrate_in_kg", True) and not source.get("nodes_only", False)
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
            ]
        )
    )

    return sum(pipelines)
