import pandas as pd
import aiohttp
import logging

from typing import List

from kedro.pipeline import Pipeline, node, pipeline
from matrix.pipelines.embeddings.nodes import _bucketize

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from matrix import settings

from . import nodes

from refit.v1.core.inject import inject_object


logger = logging.getLogger(__name__)


class NodeNormNormalizer:
    def __init__(self, conflate: bool, drug_chemical_conflate: bool) -> None:
        self._endpoint = "https://nodenormalization-sri.renci.org/get_normalized_nodes"
        self._conflate = conflate
        self._drug_chemical_conflate = drug_chemical_conflate

    async def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        curies = df["id"].tolist()

        request_json = {
            "curies": curies,
            "conflate": self._conflate,
            "drug_chemical_conflate": self._drug_chemical_conflate,
            "description": "true",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url=self._endpoint, json=request_json) as resp:
                if resp.status == 200:
                    response_json = await resp.json()
                    logger.debug(response_json)
                else:
                    logger.warning(f"Node norm response code: {resp.status}")
                    # resp_text = await resp.text()
                    resp.raise_for_status()

        df["normalized_id"] = self._extract_ids(response_json, curies)
        return df

    @staticmethod
    def _extract_ids(response, curies: List[str]):
        print(response)

        ids = []
        for curie in curies:
            el = response.get(curie, {})
            if not el:
                el = {}

            ids.append(el.get("id", {}).get("identifier", None))

        return ids


@inject_object()
def transform_nodes(nodes: DataFrame, biolink_categories, transformer):
    return transformer.transform_nodes(nodes, biolink_categories)


@inject_object()
def transform_edges(edges: DataFrame, transformer):
    return transformer.transform_edges(edges)


@inject_object()
def apply_normalization(dfs, normalizer):
    # TODO: Can we make this more generic so we avoid redefining?

    # NOTE: Inner function to avoid reference issues on unpacking
    # the dataframe, therefore leading to only the latest shard
    # being processed n times.
    def _func(dataframe: pd.DataFrame):
        return lambda df=dataframe: normalizer.normalize_df(df())

    shards = {}
    for path, df in dfs.items():
        # Little bit hacky, but extracting batch from hive partitioning for input path
        # As we know the input paths to this dataset are of the format /shard={num}
        bucket = path.split("/")[0].split("=")[1]

        # Invoke function to compute embeddings
        shard_path = f"bucket={bucket}/shard"
        shards[shard_path] = _func(df)

    return shards


def normalize(nodes, edges):
    nodes = (
        nodes.withColumn("normalization_success", F.col("normalized_id").isNotNull())
        # avoids nulls in id column, if we couldn't resolve IDs, we keep original
        .withColumn("normalized_id", F.coalesce(F.col("normalized_id"), F.col("id")))
    )

    mapping_df = nodes.select("id", "normalized_id", "normalization_success")

    # edges are bit more complex, we need to map both the subject and object
    edges = edges.join(
        mapping_df.withColumnsRenamed(
            {
                "id": "subject",
                "normalized_id": "subject_normalized",
                "normalization_success": "subject_normalization_success",
            }
        ),
        on="subject",
        how="left",
    )
    edges = edges.join(
        mapping_df.withColumnsRenamed(
            {
                "id": "object",
                "normalized_id": "object_normalized",
                "normalization_success": "object_normalization_success",
            }
        ),
        on="object",
        how="left",
    )
    edges = edges.withColumnsRenamed({"subject": "original_subject", "object": "original_object"}).withColumnsRenamed(
        {"subject_normalized": "subject", "object_normalized": "object"}
    )

    return nodes.withColumnsRenamed({"id": "original_id"}).withColumnsRenamed({"normalized_id": "id"}), edges


def _create_integration_pipeline(source: str) -> Pipeline:
    return pipeline(
        [
            node(
                func=transform_nodes,
                inputs=[
                    f"ingestion.int.{source}.nodes",
                    "integration.raw.biolink.categories",
                    f"params:integration.sources.{source}.transformer",
                ],
                outputs=f"integration.int.{source}.nodes",
                name=f"transform_{source}_nodes",
                tags=["standardize"],
            ),
            node(
                func=transform_edges,
                inputs=[f"ingestion.int.{source}.edges", f"params:integration.sources.{source}.transformer"],
                outputs=f"integration.int.{source}.edges",
                name=f"transform_{source}_edges",
                tags=["standardize"],
            ),
            node(
                func=_bucketize,
                inputs=[f"integration.int.{source}.nodes", "params:integration.batch_size"],
                outputs=f"integration.int.{source}.bucketized_nodes@spark",
                name=f"bucketize_{source}_kg",
            ),
            node(
                func=apply_normalization,
                inputs=[f"integration.int.{source}.bucketized_nodes@partitioned", "params:integration.normalizer"],
                outputs=f"integration.int.{source}.normalized_nodes@partitioned",
                name=f"normalize_{source}_kg_nodes",
            ),
            node(
                func=normalize,
                inputs={
                    "nodes": f"integration.int.{source}.normalized_nodes@spark",
                    "edges": f"integration.int.{source}.edges",
                },
                outputs=[
                    f"integration.int.{source}.nodes.norm",
                    f"integration.int.{source}.edges.norm",
                ],
                name=f"normalize_{source}_kg",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    pipelines = []
    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration"):
        pipelines.append(
            pipeline(
                _create_integration_pipeline(source=source["name"]),
                tags=[source["name"]],
            )
        )

    pipelines.append(
        pipeline(
            [
                node(
                    func=nodes.union_and_deduplicate_nodes,
                    inputs=[
                        f'integration.int.{source["name"]}.nodes.norm'
                        for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration")
                    ],
                    outputs="integration.prm.unified_nodes",
                    name="create_prm_unified_nodes",
                ),
                # union edges
                node(
                    func=nodes.union_and_deduplicate_edges,
                    inputs=[
                        f'integration.int.{source["name"]}.edges.norm'
                        for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration")
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
                        "integration.raw.biolink.predicates",
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


# def create_old_pipeline(**kwargs) -> Pipeline:
#     """Create integration pipeline."""
#     return pipeline(
#         [
#             node(
#                 func=transform_robo_nodes,
#                 inputs=["ingestion.int.robokop.nodes", "integration.raw.biolink.categories"],
#                 outputs="integration.int.robokop.nodes",
#                 name="transform_robokop_nodes",
#                 tags=["standardize"],
#             ),
#             node(
#                 func=transform_robo_edges,
#                 inputs=["ingestion.int.robokop.edges"],
#                 outputs="integration.int.robokop.edges",
#                 name="transform_robokop_edges",
#                 tags=["standardize"],
#             ),
#             node(
#                 func=transform_rtxkg2_nodes,
#                 inputs="ingestion.int.rtx_kg2.nodes",
#                 outputs="integration.int.rtx.nodes",
#                 name="transform_rtx_nodes",
#                 tags=["standardize"],
#             ),
#             node(
#                 func=transform_rtxkg2_edges,
#                 inputs=[
#                     "ingestion.int.rtx_kg2.edges",
#                     "ingestion.int.rtx_kg2.curie_to_pmids",
#                     "params:integration.preprocessing.rtx.semmed_filters",
#                 ],
#                 outputs="integration.int.rtx.edges",
#                 name="transform_rtx_edges",
#                 tags=["standardize"],
#             ),
#             # Normalize the KG IDs
#             node(
#                 func=nodes.normalize_kg,
#                 inputs={
#                     "nodes": "integration.int.rtx.nodes",
#                     "edges": "integration.int.rtx.edges",
#                     "api_endpoint": "params:integration.nodenorm.api_endpoint",
#                     "conflate": "params:integration.nodenorm.conflate",
#                     "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
#                     "batch_size": "params:integration.nodenorm.batch_size",
#                     "parallelism": "params:integration.nodenorm.parallelism",
#                 },
#                 outputs=[
#                     "integration.int.rtx.nodes.norm",
#                     "integration.int.rtx.edges.norm",
#                     "integration.int.rtx.nodes_norm_mapping",
#                 ],
#                 name="normalize_rtx_kg",
#                 tags=["standardize"],
#             ),
#             node(
#                 func=nodes.normalize_kg,
#                 inputs={
#                     "nodes": "integration.int.robokop.nodes",
#                     "edges": "integration.int.robokop.edges",
#                     "api_endpoint": "params:integration.nodenorm.api_endpoint",
#                     "conflate": "params:integration.nodenorm.conflate",
#                     "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
#                     "batch_size": "params:integration.nodenorm.batch_size",
#                     "parallelism": "params:integration.nodenorm.parallelism",
#                 },
#                 outputs=[
#                     "integration.int.robokop.nodes.norm",
#                     "integration.int.robokop.edges.norm",
#                     "integration.int.robokop.nodes_norm_mapping",
#                 ],
#                 name="normalize_robokop_kg",
#             ),
#             node(
#                 func=nodes.union_and_deduplicate_nodes,
#                 inputs={
#                     "datasets_to_union": "params:integration.unification.datasets_to_union",
#                     "rtx": "integration.int.rtx.nodes.norm",
#                     "biolink_categories_df": "integration.raw.biolink.categories",
#                     "robokop": "integration.int.robokop.nodes.norm",
#                     "medical_team": "ingestion.int.ec_medical_team.nodes",
#                 },
#                 outputs="integration.prm.unified_nodes",
#                 name="create_prm_unified_nodes",
#             ),
#             # union edges
#             node(
#                 func=nodes.union_and_deduplicate_edges,
#                 inputs={
#                     "datasets_to_union": "params:integration.unification.datasets_to_union",
#                     "rtx": "integration.int.rtx.edges.norm",
#                     "robokop": "integration.int.robokop.edges.norm",
#                     "medical_team": "ingestion.int.ec_medical_team.edges",
#                 },
#                 outputs="integration.prm.unified_edges",
#                 name="create_prm_unified_edges",
#             ),
#             # filter nodes given a set of filter stages
#             node(
#                 func=nodes.prefilter_unified_kg_nodes,
#                 inputs=[
#                     "integration.prm.unified_nodes",
#                     "params:integration.filtering.node_filters",
#                 ],
#                 outputs="integration.prm.prefiltered_nodes",
#                 name="prefilter_prm_knowledge_graph_nodes",
#                 tags=["filtering"],
#             ),
#             # filter edges given a set of filter stages
#             node(
#                 func=nodes.filter_unified_kg_edges,
#                 inputs=[
#                     "integration.prm.prefiltered_nodes",
#                     "integration.prm.unified_edges",
#                     "integration.raw.biolink.predicates",
#                     "params:integration.filtering.edge_filters",
#                 ],
#                 outputs="integration.prm.filtered_edges",
#                 name="filter_prm_knowledge_graph_edges",
#                 tags=["filtering"],
#             ),
#             node(
#                 func=nodes.filter_nodes_without_edges,
#                 inputs=[
#                     "integration.prm.prefiltered_nodes",
#                     "integration.prm.filtered_edges",
#                 ],
#                 outputs="integration.prm.filtered_nodes",
#                 name="filter_nodes_without_edges",
#                 tags=["filtering"],
#             ),
#         ]
#     )
