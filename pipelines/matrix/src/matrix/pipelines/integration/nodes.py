"""Nodes for the ingration pipeline."""

import logging
import os
from functools import partial, reduce
from typing import Any, Dict, List

import pandas as pd
import pandera.pyspark as pa
import pyspark as ps
import pyspark.sql.functions as F
import requests
from joblib import Memory
from pyspark.sql import DataFrame
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema

# TODO move these into config
memory = Memory(location=".cache/nodenorm", verbose=0)
nn_endpoint = f'{os.getenv("NODE_NORMALIZER_SERVER", "https://nodenormalization-sri.renci.org/")}/get_normalized_nodes'
logger = logging.getLogger(__name__)


@pa.check_output(KGEdgeSchema)
def unify_edges(*edges) -> DataFrame:
    """Function to unify edges datasets."""
    # Union edges
    union = reduce(partial(DataFrame.unionByName, allowMissingColumns=True), edges)

    # Deduplicate
    return KGEdgeSchema.group_edges_by_id(union)


@pa.check_output(KGNodeSchema)
def unify_nodes(*nodes) -> DataFrame:
    """Function to unify nodes datasets."""
    # Union nodes
    union = reduce(partial(DataFrame.unionByName, allowMissingColumns=True), nodes)

    # Deduplicate
    return KGNodeSchema.group_nodes_by_id(union)


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


def normalize_kg(nodes: ps.sql.DataFrame, edges: ps.sql.DataFrame):
    """Function normalizes a KG using external API endpoint.

    This functions takes the nodes and edges frames for a KG and leverages
    an external API to map the nodes to their normalized IDs.
    It returns the datasets with

    """
    node_ids = nodes.select("id").distinct().orderBy("id").toPandas()["id"].to_list()
    node_id_map = batch_map_ids(node_ids)

    # convert dict back to a dataframe to parallelize the mapping
    node_id_map_df = pd.DataFrame(list(node_id_map.items()), columns=["id", "normalized_id"])
    spark = ps.sql.SparkSession.builder.getOrCreate()
    mapping_df = spark.createDataFrame(node_id_map_df)

    # add normalized_id to nodes
    nodes = (
        nodes.join(mapping_df, on="id", how="left")
        .withColumnsRenamed({"id": "original_id"})
        .withColumnsRenamed({"normalized_id": "id"})
    )

    # edges are bit more complex, we need to map both the subject and object
    edges = edges.join(
        mapping_df.withColumnsRenamed({"id": "subject", "normalized_id": "subject_normalized"}),
        on="subject",
        how="left",
    )
    edges = edges.join(
        mapping_df.withColumnsRenamed({"id": "object", "normalized_id": "object_normalized"}),
        on="object",
        how="left",
    )
    edges = edges.withColumnsRenamed({"subject": "original_subject", "object": "original_object"}).withColumnsRenamed(
        {"subject_normalized": "subject", "object_normalized": "object"}
    )

    return nodes, edges


@retry(stop_after_attempt(3), wait_random_exponential(min=1, max=10))
@memory.cache
def hit_node_norm_service(
    curies: List[str],
    endpoint: str,
    conflate: bool = False,
    drug_chemical_conflate: bool = False,
):
    """Hits the node normalization service with a list of curies.

    Makes heavy use of tenacity for retries and joblib for caching locally on disk

    Args:
        curies (List[str]): A list of curies to normalize.
        endpoint (str): The endpoint to hit.
        conflate (bool, optional): Whether to conflate the nodes. Defaults to False.
        drug_chemical_conflate (bool, optional): Whether to conflate the drug and chemical nodes. Defaults to False.

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
    resp: requests.models.Response = requests.post(url=endpoint, json=request_json)
    logger.debug(resp.json())

    if resp.status_code == 200:
        # if successful return the json as an object
        return _extract_ids(resp.json())
    else:
        logger.warning(f"Node norm response code: {resp.status_code}")
        logger.debug(resp.text)
        resp.raise_for_status()


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


def batch_map_ids(ids: List[str], batch_size: int = 100, parallelism: int = 10):
    """Maps a list of ids to their normalized form using the node normalization service.

    Args:
        ids (List[str]): A list of ids to map.
        batch_size (int, optional): The number of ids to map in a single batch. Defaults to 100.
        parallelism (int, optional): The number of threads to use for parallel processing. Defaults to 100.

    Returns:
        Dict[str, str]: A dictionary of the form {id: normalized_id}.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    mappings = {}
    total_batches = (len(ids) + batch_size - 1) // batch_size

    def _process_batch(batch):
        return hit_node_norm_service(batch, nn_endpoint)

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = []
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            futures.append(executor.submit(_process_batch, batch))

        for future in tqdm(as_completed(futures), total=total_batches, desc="Batch Mapping IDs"):
            mappings.update(future.result())

    assert len(mappings) == len(ids)
    return mappings
