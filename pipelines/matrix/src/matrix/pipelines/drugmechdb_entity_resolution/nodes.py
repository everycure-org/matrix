"""Nodes for the DrugMechDB entity resolution pipeline."""

import pandas as pd
import logging

from typing import List, Callable
from jsonpath_ng import parse
from tqdm import tqdm

import pyspark as ps
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

from matrix.pipelines.preprocessing.nodes import resolve_name, resolve, normalize
from matrix.pipelines.integration.nodes import batch_map_ids

from refit.v1.core.inject import inject_object

logger = logging.getLogger(__name__)


def _map_name_and_curie(name: str, curie: str, endpoint: str) -> str:
    """Attempt to map an entity to the knowledge graph using either its name or curie.

    Args:
        name: The name of the node.
        curie: The curie of the node.
        endpoint: The endpoint of the synonymizer.

    Returns:
        The mapped curie. None if no mapping was found.
    """
    mapped_curie = normalize(curie, endpoint)
    if not mapped_curie:
        mapped_curie = normalize(name, endpoint)
    if not mapped_curie:
        mapped_curie = resolve(name, endpoint)

    return mapped_curie


def _map_several_ids_and_names(name: str, id_lst: List[str], synonymizer_endpoint: str) -> str:
    """Map a name and several IDs to the knowledge graph.

    Args:
        name: The name of the node.
        id_lst: List of IDs for the node.
        synonymizer_endpoint: The endpoint of the synonymizer.

    Returns:
        The mapped curie. None if no mapping was found.
    """
    id_lst = list(set(id_lst))  # Remove duplicates
    id_lst = [id for id in id_lst if id]  # Filter out Nones
    mapped_id = None
    for id in id_lst:
        mapped_id = _map_name_and_curie(name, id, synonymizer_endpoint)
        if mapped_id:
            break
    return mapped_id


def prenormalize_drugmechdb_entities_arax(drug_mech_db: List[dict], synonymizer_endpoint: str) -> pd.DataFrame:
    """Normalize DrugMechDB entities using the ARAX synonymizer.

    For drug and diseases nodes, there may be multiple IDs so we try to normalize with all of them to improve probability of mapping.

    Args:
        drug_mech_db: The DrugMechDB indication paths.
        synonymizer_endpoint: The endpoint of the synonymizer.

    Returns:
        A dataframe with the DrugMechDB ID, name, and mapped ID. If mapping fails, the mapped ID is the original ID.
    """
    df = pd.DataFrame(columns=["id", "name", "resolved_curie"])
    # Loop over all entries in the DrugMechDB
    for entry in tqdm(drug_mech_db):
        for node in entry["nodes"]:
            mapped_id = None
            # Mapping logic if node is a drug
            if node["name"] == entry["graph"]["drug"]:
                drugbank_id = entry["graph"]["drugbank"]
                drug_mesh_id = entry["graph"]["drug_mesh"]
                canonical_id = node["id"]
                mapped_id = _map_several_ids_and_names(
                    node["name"], [drugbank_id, drug_mesh_id, canonical_id], synonymizer_endpoint
                )
            # Mapping logic if node is a disease
            elif node["name"] == entry["graph"]["disease"]:
                disease_mesh_id = entry["graph"]["disease_mesh"]
                canonical_id = node["id"]
                mapped_id = _map_several_ids_and_names(
                    node["name"], [disease_mesh_id, canonical_id], synonymizer_endpoint
                )
            # Mapping logic for all other nodes
            else:
                mapped_id = _map_name_and_curie(node["name"], node["id"], synonymizer_endpoint)

            new_row = {"id": node["id"], "name": node["name"], "resolved_curie": mapped_id}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.drop_duplicates(inplace=True)
    df["resolved_curie"] = df["resolved_curie"].fillna(df["id"])
    return df


def prenormalize_drugmechdb_entities_renci(
    drug_mech_db: List[dict],
    name_resolver: str = "https://name-resolution-sri-dev.apps.renci.org",
    timeout: int = 5,
    parallelism: int = 10,
) -> pd.DataFrame:
    """Normalize DrugMechDB entities using the RENCI name resolver.

    Args:
        drug_mech_db: The DrugMechDB indication paths.
        name_resolver: The endpoint of the name resolver.
        timeout: The timeout for the name resolver.
        parallelism: Number of parallel threads to use for the name resolver.

    Returns:
        A dataframe with the DrugMechDB ID, name, and resolved curie.
    """
    # Collect DrugMechDB IDs and names
    entity_set = {(node["id"], node["name"]) for entry in drug_mech_db for node in entry["nodes"]}
    spark = ps.sql.SparkSession.builder.getOrCreate()
    nodes = spark.createDataFrame(entity_set, schema=["id", "name"])
    nodes = nodes.repartition(parallelism)

    # Try to normalise with name resolver
    resolve_name_udf = F.udf(lambda x: resolve_name(x, name_resolver, timeout=timeout), returnType=StringType())
    nodes = nodes.withColumn("resolved_curie", resolve_name_udf(F.col("name")))

    # Use curie from name resolver if available
    nodes = nodes.withColumn(
        "resolved_curie", F.when(F.col("resolved_curie").isNotNull(), F.col("resolved_curie")).otherwise(F.col("id"))
    )
    nodes_df = nodes.toPandas()
    return nodes_df


@inject_object()
def normalize_drugmechdb_entities(
    drug_mech_db: List[dict],
    prenormalize_func: Callable,
    api_endpoint: str,
    json_path_expr: str = "$.id.identifier",
    conflate: bool = True,
    drug_chemical_conflate: bool = True,
    batch_size: int = 100,
    parallelism: int = 10,
) -> pd.DataFrame:
    """Normalize DrugMechDB entities with a combination of the RENCI name resolver and Translator node normalizer.

    Args:
        drug_mech_db: List of DrugMechDB entries
        prenormalize_func: Function to prenormalize the DrugMechDB entities prior to sending through the Translator node normalizer service.
            This must be a function that takes a list of DrugMechDB entries and returns a Pandas dataframe with columns "id", "name" and "resolved_curie".
        api_endpoint: API endpoint of the translator normalization service
        json_path_expr: JSON path expression to extract the identifier from the API response
        conflate: Whether to conflate drug and chemical entities
        drug_chemical_conflate: Whether to conflate drug and chemical entities
        batch_size: Batch size for the batch map
        parallelism: Number of parallel threads to use for the batch map
    """
    # Perform prenormalization
    nodes_df = prenormalize_func(drug_mech_db)

    # Normalize with Translator
    logger.info("collecting node ids for normalization")
    node_ids = nodes_df["resolved_curie"].to_list()
    logger.info(f"collected {len(node_ids)} node ids for normalization. Performing normalization...")
    node_id_map = batch_map_ids(
        frozenset(node_ids),
        api_endpoint,
        parse(json_path_expr),
        batch_size,
        parallelism,
        conflate,
        drug_chemical_conflate,
    )
    is_na_map = {k: pd.notna(v) for k, v in node_id_map.items()}
    node_id_map = {k: v for k, v in node_id_map.items() if is_na_map.get(k, False)}
    nodes_df["resolved_curie"] = nodes_df["resolved_curie"].apply(lambda x: node_id_map.get(x, x))
    nodes_df["normalization_success"] = nodes_df["resolved_curie"].apply(lambda x: is_na_map.get(x, True))

    # Rename columns
    nodes_df = nodes_df.rename(
        columns={"resolved_curie": "mapped_ID", "name": "DrugMechDB_name", "id": "DrugMechDB_ID"}
    )

    return nodes_df
