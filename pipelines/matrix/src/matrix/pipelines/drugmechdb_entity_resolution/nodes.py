"""Nodes for the DrugMechDB entity resolution pipeline."""

import pandas as pd
import logging

from typing import List
from jsonpath_ng import parse

import pyspark as ps
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

from matrix.pipelines.preprocessing.nodes import resolve_name
from matrix.pipelines.integration.nodes import batch_map_ids

logger = logging.getLogger(__name__)


def normalize_drugmechdb_entities(
    drug_mech_db: List[dict],
    api_endpoint: str,
    name_resolver: str = "https://name-resolution-sri-dev.apps.renci.org",
    timeout: int = 5,
    json_path_expr: str = "$.id.identifier",
    conflate: bool = True,
    drug_chemical_conflate: bool = True,
    batch_size: int = 100,
    parallelism: int = 10,
) -> pd.DataFrame:
    """Normalize DrugMechDB entities with the translator.

    Args:
        drug_mech_db: List of DrugMechDB entries
        api_endpoint: API endpoint of the translator normalization service
        name_resolver: Name resolver endpoint
        timeout: Timeout for the name resolver
        json_path_expr: JSON path expression to extract the identifier from the API response
        conflate: Whether to conflate drug and chemical entities
        drug_chemical_conflate: Whether to conflate drug and chemical entities
        batch_size: Batch size for the batch map
        parallelism: Parallelism for the batch map
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

    # Normalise with Translator
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
    success_map = {k: pd.notna(v) for k, v in node_id_map.items()}
    node_id_map = {k: v for k, v in node_id_map.items() if success_map.get(v, False)}
    nodes_df["resolved_curie"] = nodes_df["resolved_curie"].apply(lambda x: node_id_map.get(x, x))
    nodes_df["normalization_success"] = nodes_df["resolved_curie"].apply(lambda x: success_map.get(x, False))

    # Rename columns
    nodes_df = nodes_df.rename(
        columns={"resolved_curie": "mapped_ID", "name": "DrugMechDB_name", "id": "DrugMechDB_ID"}
    )

    return nodes_df
