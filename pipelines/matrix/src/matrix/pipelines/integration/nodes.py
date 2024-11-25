import asyncio
import logging
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Tuple

import aiohttp
import pandas as pd
import pandera.pyspark as pa
import pyspark as ps
import pyspark.sql.functions as F
from joblib import Memory
from jsonpath_ng import parse
from more_itertools import chunked
from pyspark.sql import DataFrame
from refit.v1.core.inject import inject_object
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
)
from tqdm.asyncio import tqdm_asyncio

from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema, cols_for_schema

# TODO move these into config
memory = Memory(location=".cache/nodenorm", verbose=0)
logger = logging.getLogger(__name__)


@pa.check_output(KGEdgeSchema)
def union_and_deduplicate_edges(*edges) -> DataFrame:
    """Function to unify edges datasets."""
    # fmt: off
    return (
        _union_datasets(*edges)
        .transform(KGEdgeSchema.group_edges_by_id)
    )
    # fmt: on


@pa.check_output(KGNodeSchema)
def union_and_deduplicate_nodes(*nodes) -> DataFrame:
    """Function to unify nodes datasets."""

    # fmt: off
    return (
        _union_datasets(*nodes)

        # first we group the dataset by id to deduplicate
        .transform(KGNodeSchema.group_nodes_by_id)

        # next we need to apply a number of transformations to the nodes to ensure grouping by id did not select wrong information
        #.transform(determine_most_specific_category, biolink_categories_df) TODO: Do we still want this?

        # finally we select the columns that we want to keep
        .select(*cols_for_schema(KGNodeSchema))
    )
    # fmt: on


def _union_datasets(
    *datasets: DataFrame,
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
    return reduce(partial(DataFrame.unionByName, allowMissingColumns=True), datasets)


def _apply_transformations(
    df: DataFrame, transformations: List[Tuple[Callable, Dict[str, Any]]], **kwargs
) -> DataFrame:
    logger.info(f"Filtering dataframe with {len(transformations)} transformations")
    last_count = df.count()
    logger.info(f"Number of rows before filtering: {last_count}")
    for name, transformation in transformations.items():
        logger.info(f"Applying transformation: {name}")
        df = df.transform(transformation, **kwargs)
        new_count = df.count()
        logger.info(f"Number of rows after transformation: {new_count}, cut out {last_count - new_count} rows")
        last_count = new_count

    return df


@inject_object()
def prefilter_unified_kg_nodes(
    nodes: DataFrame,
    transformations: List[Tuple[Callable, Dict[str, Any]]],
) -> DataFrame:
    return _apply_transformations(nodes, transformations)


@inject_object()
def filter_unified_kg_edges(
    nodes: DataFrame,
    edges: DataFrame,
    biolink_predicates: Dict[str, Any],
    transformations: List[Tuple[Callable, Dict[str, Any]]],
) -> DataFrame:
    """Function to filter the knowledge graph edges.

    We first apply a series for filter transformations, and then deduplicate the edges based on the nodes that we dropped.
    No edge can exist without its nodes.
    """

    # filter down edges to only include those that are present in the filtered nodes
    edges_count = edges.count()
    logger.info(f"Number of edges before filtering: {edges_count}")
    edges = (
        edges.alias("edges")
        .join(nodes.alias("subject"), on=F.col("edges.subject") == F.col("subject.id"), how="inner")
        .join(nodes.alias("object"), on=F.col("edges.object") == F.col("object.id"), how="inner")
        .select("edges.*")
    )
    new_edges_count = edges.count()
    logger.info(f"Number of edges after filtering: {new_edges_count}, cut out {edges_count - new_edges_count} edges")

    return _apply_transformations(edges, transformations, biolink_predicates=biolink_predicates)


def filter_nodes_without_edges(
    nodes: DataFrame,
    edges: DataFrame,
) -> DataFrame:
    """Function to filter nodes without edges.

    Args:
        nodes: nodes df
        edges: edge df
    Returns"
        Final dataframe of nodes with edges
    """

    # Construct list of edges
    logger.info("Nodes before filtering: %s", nodes.count())
    edge_nodes = (
        edges.withColumn("id", F.col("subject"))
        .unionByName(edges.withColumn("id", F.col("object")))
        .select("id")
        .distinct()
    )

    nodes = nodes.alias("nodes").join(edge_nodes, on="id").select("nodes.*").persist()
    logger.info("Nodes after filtering: %s", nodes.count())
    return nodes


@memory.cache
def batch_map_ids(
    ids: frozenset[str],
    api_endpoint: str,
    json_parser: parse,
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
        async_batch_map_ids(ids, api_endpoint, json_parser, batch_size, parallelism, conflate, drug_chemical_conflate)
    )

    logger.info(f"mapped {len(results)} ids")
    empty_results = [id for id in ids if not results.get(id)]
    logger.warning(f"Endpoint did not return results for {len(empty_results)}")
    # ensuring we return a result for every input id, even if it's None
    return {id: results.get(id) for id in ids}


async def async_batch_map_ids(
    ids: frozenset[str],
    api_endpoint: str,
    json_parser: parse,
    batch_size: int,
    parallelism: int,
    conflate: bool,
    drug_chemical_conflate: bool,
) -> Dict[str, str]:
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(parallelism)
        tasks = [
            process_batch(batch, api_endpoint, json_parser, session, semaphore, conflate, drug_chemical_conflate)
            for batch in chunked(ids, batch_size)
        ]
        results = await tqdm_asyncio.gather(*tasks)

    return dict(item for sublist in results for item in sublist.items())


async def process_batch(
    batch: List[str],
    api_endpoint: str,
    json_parser: parse,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    conflate: bool,
    drug_chemical_conflate: bool,
) -> Dict[str, str]:
    async with semaphore:
        return await hit_node_norm_service(batch, api_endpoint, json_parser, session, conflate, drug_chemical_conflate)


def normalize_kg(
    nodes: ps.sql.DataFrame,
    edges: ps.sql.DataFrame,
    api_endpoint: str,
    json_path_expr: str = "$.id.identifier",
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
    json_parser = parse(json_path_expr)
    logger.info("collecting node ids for normalization")
    node_ids = nodes.select("id").orderBy("id").toPandas()["id"].to_list()
    logger.info(f"collected {len(node_ids)} node ids for normalization. Performing normalization...")
    node_id_map = batch_map_ids(
        frozenset(node_ids), api_endpoint, json_parser, batch_size, parallelism, conflate, drug_chemical_conflate
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
    json_parser: parse,
    session: aiohttp.ClientSession,
    conflate: bool = True,
    drug_chemical_conflate: bool = True,
):
    """Hits the node normalization service with a list of curies using aiohttp.

    Makes heavy use of tenacity for retries and joblib for caching locally on disk

    Args:
        curies (List[str]): A list of curies to normalize.
        endpoint (str): The endpoint to hit.
        json_parser: JSON path expression for attribute to get
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

    async with session.post(url=endpoint, json=request_json) as resp:
        if resp.status == 200:
            response_json = await resp.json()
            logger.debug(response_json)
            return _extract_ids(response_json, json_parser)
        else:
            logger.warning(f"Node norm response code: {resp.status}")
            resp_text = await resp.text()
            logger.debug(resp_text)
            resp.raise_for_status()


# NOTE: we are not taking the label that the API returns, this could actually be important. Do we want the labels/biolink types as well?
def _extract_ids(response: Dict[str, Any], json_parser: parse):
    ids = {}
    for key, item in response.items():
        logger.debug(f"Response for key {key}: {response.get(key)}")  # Log the response for each key
        try:
            ids[key] = json_parser.find(item)[0].value
        except (IndexError, KeyError):
            logger.debug(f"Not able to normalize for {key}: {item}, {json_parser}")
            ids[key] = None

    return ids
