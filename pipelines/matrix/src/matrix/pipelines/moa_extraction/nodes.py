"""Module with nodes for moa extraction."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
from refit.v1.core.inject import inject_object

from .neo4j_runners import Neo4jRunner
from .path_embeddings import OneHotEncoder, PathEmbeddingStrategy
from .path_mapping import PathMapper
from .negative_path_samplers import NegativePathSampler
from matrix.datasets.paths import KGPaths
from matrix.pipelines.modelling.nodes import _apply_splitter


def _tag_edges_between_types(
    runner: Neo4jRunner,
    type_1_lst: List[str],
    type_2_lst: List[str],
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
        runner: The Neo4j runner.
        type_1_lst: List of types for the first node.
        type_2_lst: List of types for the second node.
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
            updated = result[0][0]
            total_updated += updated
            if updated < batch_size:
                break
            if verbose:
                print(f"{int(total_updated/batch_size)} batches completed.")
        return total_updated

    # Reset the tag for all edges
    if verbose:
        print(f"Resetting tag {tag} for all edges. Batch size: {batch_size}...")
    update_relationships_in_batches(f"""
        MATCH ()-[r]-()
        WHERE r.{prefix}{tag} IS NULL OR r.{prefix}{tag} = true
        WITH r LIMIT {batch_size}
        SET r.{prefix}{tag} = false
        RETURN count(r)
    """)
    # Set the tag for the specific edges
    for type_1 in type_1_lst:
        for type_2 in type_2_lst:
            if verbose:
                print(f"Setting tags for {type_1} to {type_2}...")
            update_relationships_in_batches(f"""
                MATCH (n1: {type_1})-[r]-(n2: {type_2})
                WHERE r.{prefix}{tag} = false
                WITH r LIMIT {batch_size}
                SET r.{prefix}{tag} = true
                RETURN count(r)
            """)


@inject_object()
def add_tags(
    runner: Neo4jRunner,
    drug_types: List[str],
    disease_types: List[str],
    batch_size: int,
    verbose: bool,
    prefix: str = "_moa_extraction_",
) -> None:
    """Add tags to the Neo4j database.

    Args:
        runner: The Neo4j runner.
        drug_types: List of KG node types representing drugs.
        disease_types: List of KG node types representing diseases.
        batch_size: The batch size to use for the query.
        verbose: Whether to print the number of batches completed.
        prefix: The prefix to add to the tag.
    """
    _tag_edges_between_types(runner, drug_types, disease_types, "drug_disease", batch_size, verbose, prefix)
    _tag_edges_between_types(runner, disease_types, disease_types, "disease_disease", batch_size, verbose, prefix)


@inject_object()
def get_one_hot_encodings(runner: Neo4jRunner) -> Tuple[OneHotEncoder, OneHotEncoder]:
    """Get the one-hot encodings for node categories and edge relations .

    Args:
        runner: The Neo4j runner.

    Returns:
        A tuple of OneHotEncoder objects for node categories and edge relations.
    """
    # Get the node categories
    result = runner.run("""
        MATCH (n)
        RETURN DISTINCT n.category
    """)
    # Flatten because Neo4j result is a list of lists of the form [[category]]
    node_categories = [item for sublist in result for item in sublist]
    # Get the edge relations
    result = runner.run("""
        MATCH ()-[r]-()
        RETURN DISTINCT type(r)
    """)
    # Neo4j result is a list of lists of the form [relation]
    edge_relations = [relation[0] for relation in result]

    # Create the one-hot encoders
    category_encoder = OneHotEncoder(node_categories)
    relation_encoder = OneHotEncoder(edge_relations)
    return category_encoder, relation_encoder


@inject_object()
def map_drug_mech_db(
    runner: Neo4jRunner,
    drug_mech_db: Dict[str, Any],
    mapper: PathMapper,
    synonymizer_endpoint: str,
) -> KGPaths:
    """Map the DrugMechDB indication paths to 2 and 3-hop paths in the graph.

    Args:
        runner: The Neo4j runner.
        drug_mech_db: The DrugMechDB indication paths.
        mapper: Strategy for mapping paths to the graph.
        synonymizer_endpoint: The endpoint of the synonymizer.
    """
    paths = mapper.run(runner, drug_mech_db, synonymizer_endpoint)
    return paths.df


@inject_object()
def make_splits(
    paths_data: KGPaths,
    splitter: BaseCrossValidator,
) -> pd.DataFrame:
    """Function to split a paths dataset with

    Args:
        paths_data: Knowledge graphs paths dataset.
        splitter: sklearn splitter object used to create train/test splits.

    Returns:
        Paths dataset with split information added.
    """
    df = paths_data.df
    df_splits = _apply_splitter(df, splitter)
    return KGPaths(df=df_splits).df


@inject_object()
def generate_negative_paths(
    paths: KGPaths,
    negative_sampler_list: List[NegativePathSampler],
    runner: Neo4jRunner,
) -> KGPaths:
    """Enrich a dataset of positive indication paths with negative samples.

    Args:
        paths: Dataset of positive indication paths with splits information.
        negative_sampler_list: List of negative path samplers.
        runner: The Neo4j runner.
    """
    for split in ["TRAIN", "TEST"]:
        for negative_sampler in negative_sampler_list:
            paths_split = KGPaths(df=paths.df[paths.df["split"] == split])
            negative_paths = negative_sampler.run(paths_split, runner)
            negative_paths.df["split"] = split
            negative_paths.df["y"] = 0
            paths.df = pd.concat([paths.df, negative_paths.df])

    return paths.df


@inject_object()
def train_model(
    model: BaseEstimator,  # TODO: Replace with tuner
    paths: KGPaths,
    path_embedding_strategy: PathEmbeddingStrategy,
    category_encoder: OneHotEncoder,
    relation_encoder: OneHotEncoder,
) -> BaseEstimator:
    """Train the model on the entire paths dataset provided.

    Args:
        model: The model to train.
        paths: The paths dataset with a "y" column.
        path_embedding_strategy: Path embedding strategy.
        category_encoder: One-hot encoder for node categories.
        relation_encoder: One-hot encoder for edge relations.
    """
    X = path_embedding_strategy.run(paths, category_encoder, relation_encoder)
    y = paths.df["y"].to_numpy()
    X = np.reshape(X, (X.shape[0], -1))  # TODO: Remove when transformers are implemented
    # TODO: Add hyperparameter tuning here
    return model.fit(X, y)


@inject_object()
def train_model_split(
    model: BaseEstimator,  # TODO: Replace with tuner
    paths: KGPaths,
    path_embedding_strategy: PathEmbeddingStrategy,
    category_encoder: OneHotEncoder,
    relation_encoder: OneHotEncoder,
) -> BaseEstimator:
    """Train the model on the training portion of the paths dataset only.

    Args:
        model: The model to train.
        paths: The paths dataset with a "y" column and split information.
        path_embedding_strategy: Path embedding strategy.
        category_encoder: One-hot encoder for node categories.
        relation_encoder: One-hot encoder for edge relations.
    """
    paths_train = KGPaths(df=paths.df[paths.df["split"] == "TRAIN"])
    return train_model(model, paths_train, path_embedding_strategy, category_encoder, relation_encoder)
