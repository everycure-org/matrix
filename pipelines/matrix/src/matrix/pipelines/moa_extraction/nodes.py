"""Module with nodes for moa extraction."""

import pandas as pd
import json
import logging

from typing import List, Tuple

from pyspark.sql import DataFrame

from sklearn.model_selection import BaseCrossValidator

from refit.v1.core.inject import inject_object

from .path_embeddings import OneHotEncoder
from .path_mapping import PathMapper
from matrix.pipelines.moa_extraction.utils import GraphDB
from matrix.datasets.paths import KGPaths
from matrix.pipelines.modelling.nodes import apply_splitter

logger = logging.getLogger(__name__)


def _tag_edges_between_types(
    runner: GraphDB,
    types_for_first_node: List[str],
    types_for_second_node: List[str],
    tag: str,
    batch_size: int,
    verbose: bool,
    prefix: str = "_moa_extraction_",
) -> None:
    """Tag edges between two types.

    Note that this function is agnostic of directionality, that is, it will
    add the tag to an edge from the first to the second node regardless of
    the direction.

    Args:
        runner: The GraphDB object representing the KG..
        types_for_first_node: List of types for the first node.
        types_for_second_node: List of types for the second node.
        tag: The tag to add.
        batch_size: The batch size to use for the query.
        verbose: Whether to print the number of batches completed.
        prefix: The prefix to add to the tag.
    """

    def update_relationships_in_batches(query_with_limit: str):
        """Update relationships in batches.

        Args:
            query_with_limit: The query to run with a limit and a return statement for the count of edges updated.
        """
        total_updated = 0
        while True:
            result = runner.run(query_with_limit)
            updated = result[0]["count"]  # [0]
            total_updated += updated
            if updated < batch_size:
                break
            if verbose:
                logger.info(f"{int(total_updated/batch_size)} batches completed.")
        return total_updated

    # Reset the tag for all edges
    if verbose:
        logger.info(f"Resetting tag {tag} for all edges. Batch size: {batch_size}...")
    update_relationships_in_batches(f"""
        MATCH ()-[r]-()
        WHERE r.{prefix}{tag} IS NULL OR r.{prefix}{tag} = true
        WITH r LIMIT {batch_size}
        SET r.{prefix}{tag} = false
        RETURN count(r) AS count
    """)
    # Set the tag for the specific edges
    for type_1 in types_for_first_node:
        for type_2 in types_for_second_node:
            if verbose:
                logger.info(f"Setting tags for {type_1} to {type_2}...")
            update_relationships_in_batches(f"""
                MATCH (n1: {type_1})-[r]-(n2: {type_2})
                WHERE r.{prefix}{tag} = false
                WITH r LIMIT {batch_size}
                SET r.{prefix}{tag} = true
                RETURN count(r) AS count
            """)


@inject_object()
def add_tags(
    runner: GraphDB,
    drug_types: List[str],
    disease_types: List[str],
    batch_size: int,
    verbose: bool,
    edges_dummy: DataFrame,
    prefix: str = "_moa_extraction_",
) -> None:
    """Add tags to the Neo4j database.

    Args:
        runner: The GraphDB object representing the KG.
        drug_types: List of KG node types representing drugs.
        disease_types: List of KG node types representing diseases.
        batch_size: The batch size to use for the query.
        verbose: Whether to print the number of batches completed.
        edges_dummy: Dummy variable ensuring that the node is run after edges have been added to the KG.
        prefix: The prefix to add to the tag.
    """
    _tag_edges_between_types(runner, drug_types, disease_types, "drug_disease", batch_size, verbose, prefix)
    _tag_edges_between_types(runner, drug_types, drug_types, "drug_drug", batch_size, verbose, prefix)
    _tag_edges_between_types(runner, disease_types, disease_types, "disease_disease", batch_size, verbose, prefix)
    return {"success": True}


@inject_object()
def get_one_hot_encodings(
    runner: GraphDB,
    tags_dummy: DataFrame,
) -> Tuple[OneHotEncoder, OneHotEncoder]:
    """Get the one-hot encodings for node categories and edge relations.

    Args:
        runner: The GraphDB object representing the KG.
        tags_dummy: Dummy variable ensuring pipeline is run in linear order.

    Returns:
        A tuple of OneHotEncoder objects for node categories and edge relations.
    """
    # Get the node categories
    result = runner.run("""
        MATCH (n)
        RETURN DISTINCT n.category AS category
    """)
    # Flatten because Neo4j result is a list of dicts of the form [{"category" : <category>}]
    node_categories = [item["category"] for item in result]
    # Get the edge relations
    result = runner.run("""
        MATCH ()-[r]-()
        RETURN DISTINCT type(r) AS relation
    """)
    # Neo4j result is a list of dicts of the form [{"relation" : <relation>}]
    edge_relations = [relation["relation"] for relation in result]

    # Create the one-hot encoders
    category_encoder = OneHotEncoder(node_categories)
    relation_encoder = OneHotEncoder(edge_relations)

    return category_encoder, relation_encoder


@inject_object()
def map_drug_mech_db(
    runner: GraphDB,
    drug_mech_db: List[dict],
    mapper: PathMapper,
    drugmechdb_entities: pd.DataFrame,
    one_hot_encodings_dummy: dict,
) -> KGPaths:
    """Map the DrugMechDB indication paths to 2 and 3-hop paths in the graph.

    Args:
        runner: The GraphDB object representing the KG.
        drug_mech_db: The DrugMechDB indication paths.
        mapper: Strategy for mapping paths to the graph.
        drugmechdb_entities: The normalized DrugMechDB entities.
        one_hot_encodings_dummy: Dummy variable ensuring pipeline is run in linear order.
    """
    paths = mapper.run(runner, drug_mech_db, drugmechdb_entities)
    return paths.df


def report_mapping_success(
    drugmechdb_entities: pd.DataFrame,
    drug_mech_db: List[dict],
    mapped_paths: KGPaths,
) -> dict:
    """Report the success of the path mapping.

    Args:
        drugmechdb_entities: The normalized DrugMechDB entities.
        drug_mech_db: The DrugMechDB indication paths.
        mapped_paths: The mapped paths.

    Returns:
        Dictionary reporting the success of the path mapping.
    """
    report = dict()
    report["pairs_in_drugmechdb"] = len(drug_mech_db)
    report["pairs_in_mapped_paths"] = len(mapped_paths.get_unique_pairs())
    report["proportion_pairs_with_mapped_paths"] = (
        len(mapped_paths.get_unique_pairs()) / len(drug_mech_db) if len(drug_mech_db) > 0 else 0
    )
    return json.loads(json.dumps(report, default=float))


@inject_object()
def make_splits(
    paths_data: KGPaths,
    splitter: BaseCrossValidator,
    mapping_report_dummy: dict,
) -> pd.DataFrame:
    """Function to split a paths dataset.

    Args:
        paths_data: Knowledge graphs paths dataset.
        splitter: sklearn splitter object used to create train/test splits.
        mapping_report_dummy: Dummy variable ensuring pipeline is run in linear order.

    Returns:
        Paths dataset with split information added.
    """
    df = paths_data.df
    if df.empty:
        raise ValueError("Paths dataframe is empty")
    df_splits = apply_splitter(df, splitter)
    return KGPaths(df=df_splits).df
