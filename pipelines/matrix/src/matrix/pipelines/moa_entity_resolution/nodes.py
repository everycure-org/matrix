"""Nodes for the MOA entity resolution pipeline."""

import pandas as pd
import logging

from typing import List, Callable
from jsonpath_ng import parse

from matrix.pipelines.integration.nodes import batch_map_ids

from refit.v1.core.inject import inject_object

logger = logging.getLogger(__name__)


def normalize_df_translator(
    df: pd.DataFrame,
    api_endpoint: str,
    curie_column: str = "curie",
    return_column: str = "normalized_curie",
    json_path_expr: str = "$.id.identifier",
    conflate: bool = True,
    drug_chemical_conflate: bool = True,
    batch_size: int = 100,
    parallelism: int = 10,
) -> pd.DataFrame:
    """Normalize a dataframe of containing "curies" with the Translator node normalizer service.

    Args:
        df: Dataframe of entities to normalize.
        api_endpoint: API endpoint of the translator normalization service
        curie_column: Column containing the curies to normalize
        return_column: Column to store the normalized curies in
        json_path_expr: JSON path expression to extract the identifier from the API response
        conflate: Whether to conflate drug and chemical entities
        drug_chemical_conflate: Whether to conflate drug and chemical entities
        batch_size: Batch size for the batch map
        parallelism: Number of parallel threads to use for the batch map

    Returns:
        A dataframe with the original curies and the normalized curies. A column for the normalization success is also added.
        If the curie is not in the mapping, the unmapped curie is used.
    """
    logger.info("collecting node ids for normalization")
    node_ids = df[curie_column].to_list()
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
    df[return_column] = df[curie_column].apply(lambda x: node_id_map.get(x, x))
    df["normalization_success"] = df[curie_column].apply(lambda x: is_na_map.get(x, True))

    return df


@inject_object()
def normalize_drugmechdb_entities(
    drug_mech_db: List[dict],
    prenormalize_func: Callable,
    api_endpoint: str,
) -> pd.DataFrame:
    """Normalize DrugMechDB entities with a combination of the RENCI name resolver and Translator node normalizer.

    Args:
        drug_mech_db: List of DrugMechDB entries
        prenormalize_func: Function to prenormalize the DrugMechDB entities prior to sending through the Translator node normalizer service.
            This must be a function that takes a list of DrugMechDB entries and returns a Pandas dataframe with columns "id", "name" and "resolved_curie".
        api_endpoint: API endpoint of the Translator node normalizer service
    """
    # Perform prenormalization
    nodes_df = prenormalize_func(drug_mech_db)

    # Normalize with Translator
    nodes_df = normalize_df_translator(nodes_df, api_endpoint, curie_column="resolved_curie", return_column="mapped_ID")

    # Rename columns
    nodes_df = nodes_df.rename(columns={"name": "DrugMechDB_name", "id": "DrugMechDB_ID"})

    return nodes_df


def normalize_input_pairs(
    input_pairs: pd.DataFrame,
    api_endpoint: str,
) -> pd.DataFrame:
    """Normalize the input drug-disease pairs for MOA prediction.

    Args:
        input_pairs: Dataframe of drug-disease pairs
        api_endpoint: API endpoint of the Translator node normalizer service
    """
    # Normalize the drug IDs
    input_pairs = normalize_df_translator(
        input_pairs,
        api_endpoint,
        curie_column="drug_id",
        return_column="drug_id_normalized",
    )

    # Normalize the disease IDs
    input_pairs = normalize_df_translator(
        input_pairs,
        api_endpoint,
        curie_column="disease_id",
        return_column="disease_id_normalized",
    )

    input_pairs.drop(columns=["normalization_success"], inplace=True)
    return input_pairs
