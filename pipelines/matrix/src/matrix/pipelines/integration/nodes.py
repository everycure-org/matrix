"""Nodes for the ingration pipeline."""

import asyncio
import logging
from functools import partial, reduce
from typing import Any, Callable, Dict, List

import aiohttp
import pandas as pd
import pandera.pyspark as pa
import pyspark as ps
import pyspark.sql.functions as F
from joblib import Memory
from more_itertools import chunked
from pyspark.sql import DataFrame
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
)

from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema

# TODO move these into config
memory = Memory(location=".cache/nodenorm", verbose=0)
logger = logging.getLogger(__name__)


@pa.check_output(KGEdgeSchema)
def union_edges(datasets_to_union: List[str], **edges) -> DataFrame:
    """Function to unify edges datasets."""
    return _union_datasets(
        datasets_to_union,
        schema_group_by_id=KGEdgeSchema.group_edges_by_id,
        **edges,
    )


@pa.check_output(KGNodeSchema)
def union_nodes(datasets_to_union: List[str], **nodes) -> DataFrame:
    """Function to unify nodes datasets."""
    return _union_datasets(
        datasets_to_union,
        schema_group_by_id=KGNodeSchema.group_nodes_by_id,
        **nodes,
    )


def _union_datasets(
    datasets_to_union: List[str],
    schema_group_by_id: Callable[[DataFrame], DataFrame],
    **datasets: DataFrame,
) -> DataFrame:
    """
    Helper function to unify datasets and deduplicate them.

    Args:
        datasets_to_union: List of dataset names to unify.
        **datasets: Arbitrary number of DataFrame keyword arguments.
        schema_group_by_id: Function to deduplicate the unified DataFrame.

    Returns:
        A unified and deduplicated DataFrame.
    """
    selected_dfs = [datasets[name] for name in datasets_to_union if name in datasets]
    union = reduce(partial(DataFrame.unionByName, allowMissingColumns=True), selected_dfs)
    return schema_group_by_id(union)


@has_schema(
    schema={
        "label": "string",
        "source_id": "string",
        "target_id": "string",
        "property_keys": "array<string>",
        "property_values": "array<numeric>",
    },
    allow_subset=True,
)
@primary_key(primary_key=["source_id", "target_id", "label"])
def create_treats(nodes: DataFrame, df: DataFrame):
    """Function to construct treats relatonship.

    NOTE: This function requires the nodes dataset, as the nodes should be
    written _prior_ to the relationships.

    Args:
        nodes: nodes dataset
        df: Ground truth dataset
    """
    return (
        df.withColumn("label", F.when(F.col("y") == 1, "TREATS").otherwise("NOT_TREATS"))
        .withColumn(
            "properties",
            F.create_map(
                F.lit("treats"),
                F.col("y"),
            ),
        )
        .withColumn("source_id", F.col("source"))
        .withColumn("target_id", F.col("target"))
        .withColumn("property_keys", F.map_keys(F.col("properties")))
        .withColumn("property_values", F.map_values(F.col("properties")))
    )


@memory.cache
def batch_map_ids(
    ids: frozenset[str],
    api_endpoint: str,
    batch_size: int,
    parallelism: int,
    conflate: bool,
    drug_chemical_conflate: bool,
) -> Dict[str, str]:
    """Maps a list of ids to their normalized form using the node normalization service.

    This function is a synchronous wrapper around an asynchronous operation.

    Args:
        ids: A frozenset of ids to map.
        api_endpoint: The endpoint to hit for normalization.
        batch_size: The number of ids to map in a single batch.
        parallelism: The number of concurrent requests to make.
        conflate: Whether to conflate the nodes.
        drug_chemical_conflate: Whether to conflate the drug and chemical nodes.

    Returns:
        Dict[str, str]: A dictionary of the form {id: normalized_id}.
    """
    results = asyncio.run(
        async_batch_map_ids(ids, api_endpoint, batch_size, parallelism, conflate, drug_chemical_conflate)
    )

    logger.info(f"mapped {len(results)} ids")
    logger.warning(f"Endpoint did not return results for {len(ids) - len(results)} ids")
    # ensuring we return a result for every input id, even if it's None
    return {id: results.get(id) for id in ids}


async def async_batch_map_ids(
    ids: frozenset[str],
    api_endpoint: str,
    batch_size: int,
    parallelism: int,
    conflate: bool,
    drug_chemical_conflate: bool,
) -> Dict[str, str]:
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(parallelism)
        tasks = [
            process_batch(batch, api_endpoint, session, semaphore, conflate, drug_chemical_conflate)
            for batch in chunked(ids, batch_size)
        ]
        results = await asyncio.gather(*tasks)

    return dict(item for sublist in results for item in sublist.items())


async def process_batch(
    batch: List[str],
    api_endpoint: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    conflate: bool,
    drug_chemical_conflate: bool,
) -> Dict[str, str]:
    async with semaphore:
        return await hit_node_norm_service(batch, api_endpoint, session, conflate, drug_chemical_conflate)


def normalize_kg(
    nodes: ps.sql.DataFrame,
    edges: ps.sql.DataFrame,
    api_endpoint: str,
    conflate: bool = True,
    drug_chemical_conflate: bool = True,
    batch_size: int = 100,
    parallelism: int = 10,
) -> ps.sql.DataFrame:
    """Function normalizes a KG using external API endpoint.

    This function takes the nodes and edges frames for a KG and leverages
    an external API to map the nodes to their normalized IDs.
    It returns the datasets with normalized IDs.

    """
    logger.info("collecting node ids for normalization")
    node_ids = nodes.select("id").orderBy("id").toPandas()["id"].to_list()
    logger.info(f"collected {len(node_ids)} node ids for normalization. Performing normalization...")
    node_id_map = batch_map_ids(
        frozenset(node_ids), api_endpoint, batch_size, parallelism, conflate, drug_chemical_conflate
    )

    # convert dict back to a dataframe to parallelize the mapping
    node_id_map_df = pd.DataFrame(list(node_id_map.items()), columns=["id", "normalized_id"])
    spark = ps.sql.SparkSession.builder.getOrCreate()
    mapping_df = (
        spark.createDataFrame(node_id_map_df)
        .withColumn("normalization_success", F.col("normalized_id").isNotNull())
        # avoids nulls in id column, if we couldn't resolve IDs, we keep original
        .withColumn("normalized_id", F.coalesce(F.col("normalized_id"), F.col("id")))
    )

    # add normalized_id to nodes
    nodes = (
        nodes.join(mapping_df, on="id", how="left")
        .withColumnsRenamed({"id": "original_id"})
        .withColumnsRenamed({"normalized_id": "id"})
    )

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

    return nodes, edges, mapping_df


@retry(
    wait=wait_exponential(min=1, max=30),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    before_sleep=print,
)
async def hit_node_norm_service(
    curies: List[str],
    endpoint: str,
    session: aiohttp.ClientSession,
    conflate: bool = True,
    drug_chemical_conflate: bool = True,
):
    """Hits the node normalization service with a list of curies using aiohttp.

    Makes heavy use of tenacity for retries and joblib for caching locally on disk

    Args:
        curies (List[str]): A list of curies to normalize.
        endpoint (str): The endpoint to hit.
        session (aiohttp.ClientSession): The aiohttp session to use for requests.
        conflate (bool, optional): Whether to conflate the nodes.
        drug_chemical_conflate (bool, optional): Whether to conflate the drug and chemical nodes.

    Returns:
        Dict[str, str]: A dictionary of the form {id: normalized_id}.

    """
    request_json = {
        "curies": curies,
        "conflate": conflate,
        "drug_chemical_conflate": drug_chemical_conflate,
        "description": "true",
    }

    logger.debug(request_json)
    async with session.post(url=endpoint, json=request_json) as resp:
        if resp.status == 200:
            response_json = await resp.json()
            logger.debug(response_json)
            return _extract_ids(response_json)
        else:
            logger.warning(f"Node norm response code: {resp.status}")
            resp_text = await resp.text()
            logger.debug(resp_text)
            resp.raise_for_status()


# NOTE: we are not taking the label that the API returns, this could actually be important. Do we want the labels/biolink types as well?
def _extract_ids(response: Dict[str, Any]):
    ids = {}
    for k in response:
        try:
            if response[k] is None:
                ids[k] = None
            else:
                ids[k] = response[k]["id"]["identifier"]
        except KeyError:
            logger.warning(f"KeyError for {k}: {response[k]}")
            ids[k] = None
    return ids
