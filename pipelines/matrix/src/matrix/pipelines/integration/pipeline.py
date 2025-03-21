from kedro.pipeline import Pipeline, node, pipeline

from matrix import settings
from matrix.pipelines.batch import pipeline as batch_pipeline

from ...kedro4argo_node import ArgoNode, ArgoResourceConfig
from . import nodes


# this pipeline is created here for faster debugging locally
# `kedro run -p cached_normalization` will run this pipeline
# TODO: clean it up, also delete it from the pipeline_registry
def create_cached_normalization_pipeline(**kwargs) -> Pipeline:
    pipelines = []
    # Create pipeline per source
    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration"):
        source = source["name"]
        pipelines.append(
            batch_pipeline.cached_api_enrichment_pipeline(
                source=f"normalization_source_{source}",
                workers=20,
                input=f"integration.int.{source}.nodes",
                output=f"integration.int.{source}.nodes_norm_mapping",
                preprocessor="params:integration.normalization.preprocessor",
                cache_miss_resolver="params:integration.normalization.normalizer",
                api="params:integration.normalization.api",
                new_col="params:integration.normalization.target_col",
                primary_key="params:integration.normalization.primary_key",
                batch_size="params:integration.normalization.batch_size",
                cache_schema="params:integration.normalization.cache_schema",
            ),
        )
    return sum(pipelines)


def _create_integration_pipeline(source: str, has_nodes: bool = True, has_edges: bool = True) -> Pipeline:
    pipelines = []

    pipelines.append(
        pipeline(
            [
                ArgoNode(
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
                    argo_config=ArgoResourceConfig(
                        memory_request=128,
                        memory_limit=128,
                    ),
                ),
                batch_pipeline.cached_api_enrichment_pipeline(
                    source=f"normalization_source_{source}",
                    workers=20,
                    input=f"integration.int.{source}.nodes",
                    output=f"integration.int.{source}.nodes_norm_mapping",
                    preprocessor="params:integration.normalization.preprocessor",
                    cache_miss_resolver="params:integration.normalization.normalizer",
                    api="params:integration.normalization.api",
                    new_col="params:integration.normalization.target_col",
                    primary_key="params:integration.normalization.primary_key",
                    batch_size="params:integration.normalization.batch_size",
                    cache_schema="params:integration.normalization.cache_schema",
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
                ArgoNode(
                    func=nodes.union_edges,
                    inputs=[
                        f'integration.int.{source["name"]}.edges.norm@spark'
                        for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration")
                        if source.get("integrate_in_kg", True)
                    ],
                    outputs="integration.prm.unified_edges",
                    name="create_prm_unified_edges",
                    argo_config=ArgoResourceConfig(memory_request=72, memory_limit=72),
                ),
            ]
        )
    )

    return sum(pipelines)
