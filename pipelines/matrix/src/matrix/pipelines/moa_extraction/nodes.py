"""Module with nodes for moa extraction."""

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
from neo4j import GraphDatabase

from refit.v1.core.inject import inject_object

from .path_embeddings import OneHotEncoder, PathEmbeddingStrategy
from .path_mapping import PathMapper
from .path_generators import PathGenerator
from matrix.datasets.paths import KGPaths
from matrix.pipelines.modelling.nodes import _apply_splitter


class Neo4jRunner:
    """Helper class for running neo4j queries."""

    def __init__(self, uri: str, user: str, password: str, database: str):
        """Initialize the Neo4j runner.

        Args:
            uri: The URI of the Neo4j instance.
            user: The user to connect to the Neo4j instance.
            password: The password to connect to the Neo4j instance.
            database: The name of the database containing the knowledge graph.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password), database=database)

    def run(self, query: str):
        """Run a query on the Neo4j database.

        Args:
            query: The query to run.
        """
        with self.driver.session() as session:
            info = session.run(query)
            x = info.values()
        return x


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
    negative_sampler_list: List[PathGenerator],
    runner: Neo4jRunner,
) -> KGPaths:
    """Enrich a dataset of positive indication paths with negative samples.

    Args:
        paths: Dataset of positive indication paths with splits information.
        negative_sampler_list: List of path generators for negative path samplers.
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
    model: BaseEstimator,
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
    X = X.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)
    return model.fit(X, y)


@inject_object()
def train_model_split(
    model: BaseEstimator,
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


def _make_predictions(
    model: BaseEstimator,
    runner: Neo4jRunner,
    pairs: pd.DataFrame,
    path_generator: PathGenerator,
    path_embedding_strategy: PathEmbeddingStrategy,
    category_encoder: OneHotEncoder,
    relation_encoder: OneHotEncoder,
) -> Dict[str, KGPaths]:
    """Make MOA predictions on the pairs dataset.

    Args:
        model: The model to make predictions with.
        pairs: Dataset of drug-disease pairs. Expected columns: (source_id, target_id).
        path_generator: Path generator outputting all paths of interest between a given drug disease pair.
        path_embedding_strategy: Path embedding strategy.
        category_encoder: One-hot encoder for node categories.
        relation_encoder: One-hot encoder for edge relations.

    Returns:
        Dictionary of KGPaths objects, one for each pair. Each object contains a dataframe with the paths sorted by confidence score.
    """
    paths_dict = dict()
    for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
        # Generate all paths for the pair
        all_paths_for_pair = path_generator.run(runner, row.source_id, row.target_id)

        # Embed the paths
        X = path_embedding_strategy.run(all_paths_for_pair, category_encoder, relation_encoder)
        X = X.astype(np.float32)

        # Make predictions
        y_pred = model.predict_proba(X)[:, 1]  # Returns column vector
        y_pred = y_pred.reshape(-1, 1)

        # Add predictions to the paths dataframe and sort
        all_paths_for_pair.df["y_pred"] = y_pred
        all_paths_for_pair.df = all_paths_for_pair.df.sort_values(by="y_pred", ascending=False)
        all_paths_for_pair.df = all_paths_for_pair.df.reset_index(drop=True)
        paths_dict["__".join((row.source_id, row.target_id))] = all_paths_for_pair

    # Return KGPaths objects as DataFrames
    paths_dict = {k: v.df for k, v in paths_dict.items()}
    return paths_dict


def _give_test_pairs(positive_paths: KGPaths) -> pd.DataFrame:
    """Get the test pairs from the positive paths dataset.

    Args:
        positive_paths: Dataset of positive indication paths, with a "split" information.

    Returns:
        DataFrame of drug-disease pairs.
    """
    test_paths = KGPaths(df=positive_paths.df[positive_paths.df["split"] == "TEST"])
    return test_paths.get_unique_pairs()


@inject_object()
def make_evaluation_predictions(
    model: BaseEstimator,
    runner: Neo4jRunner,
    positive_paths: KGPaths,
    path_generator: PathGenerator,
    path_embedding_strategy: PathEmbeddingStrategy,
    category_encoder: OneHotEncoder,
    relation_encoder: OneHotEncoder,
) -> Dict[str, KGPaths]:
    """Make MOA predictions on the pairs dataset.

    Args:
        model: The model to make predictions with.
        runner: The Neo4j runner.
        positive_paths: Dataset of positive indication paths, with a "split" information.
        path_generator: Path generator outputting all paths of interest between a given drug-disease pair.
        path_embedding_strategy: Path embedding strategy.
        category_encoder: One-hot encoder for node categories.
        relation_encoder: One-hot encoder for edge relations.

    Returns:
        Dictionary of KGPaths objects, one for each pair.
    """
    test_pairs = _give_test_pairs(positive_paths)
    return _make_predictions(
        model, runner, test_pairs, path_generator, path_embedding_strategy, category_encoder, relation_encoder
    )


def _label_evaluation_predictions(
    positive_paths: KGPaths,
    predictions_for_pair: KGPaths,
) -> KGPaths:
    """Label known test positive paths in the predictions.

    Additionally, any known positive paths in the training set are removed from the predictions.

    Args:
        positive_paths: Dataset of positive indication paths, with a "split" information.
        predictions_for_pair: KGPaths object with MOA predictions for a given pair.

    Returns:
        MOA predictions labeled as positive (y=1) or negative (y=0).
    """
    # Get the columns for the paths dataframe
    num_hops = positive_paths.num_hops
    if num_hops != predictions_for_pair.num_hops:
        raise ValueError("Number of hops in positive paths and predictions for pair do not match")
    cols = KGPaths.get_columns(num_hops)

    # Split the positive paths into test and train
    test_paths = positive_paths.df[positive_paths.df["split"] == "TEST"][cols]
    train_paths = positive_paths.df[positive_paths.df["split"] != "TEST"][cols]

    # Convert paths to tuples for fast comparison
    test_paths_set = set(map(tuple, test_paths.values))
    train_paths_set = set(map(tuple, train_paths.values))

    # Remove any paths that are in the training set and label the remaining paths
    paths_tuples = predictions_for_pair.df[cols].apply(tuple, axis=1)

    in_train = ~paths_tuples.isin(train_paths_set)
    predictions_for_pair.df = predictions_for_pair.df[in_train]

    in_test = paths_tuples[in_train].isin(test_paths_set)
    predictions_for_pair.df["y"] = in_test.astype(int)

    return predictions_for_pair


def compute_evaluation_metrics(
    positive_paths: KGPaths,
    predictions: Dict[str, KGPaths],
    k_lst: List[int],
) -> Dict[str, float]:
    """Compute the evaluation metrics for the predictions.

    Args:
        positive_paths: Dataset of positive indication paths, with a "split" information.
        predictions: Dictionary of KGPaths objects, one for each pair.
            Each KGPaths object represents the MOA predictions sorted by the confidence score.
        k_lst: List of values for Hit@k.

    Returns:
        Dictionary of evaluation metrics:
            - Hit@k, one for each value of k.
            - MRR
    """
    test_pairs = _give_test_pairs(positive_paths)

    # Compute list representing the lowest rank of a positive path for each pair
    rank_lst = []
    for _, row in test_pairs.iterrows():
        drug = row.source_id
        disease = row.target_id
        predictions_load_func = predictions[f"{drug}__{disease}"]
        predictions_for_pair = predictions_load_func()
        predictions_for_pair.df = predictions_for_pair.df.reset_index(drop=True)
        labelled_predictions = _label_evaluation_predictions(positive_paths, predictions_for_pair)
        is_positive = labelled_predictions.df["y"].eq(1)
        if is_positive.any():
            positive_ranks = labelled_predictions.df[is_positive].index + 1
            rank_lst.append(positive_ranks[0])

    # Compute evaluation metrics
    report = dict()
    rank_arr = np.array(rank_lst)
    for k in k_lst:
        hit_at_k = (rank_arr <= k).mean()
        report[f"Hit@{k}"] = hit_at_k
    report["MRR"] = (1 / rank_arr).mean()

    return report


# def make_reporting_predictions(
#     model: BaseEstimator,
#     runner: Neo4jRunner,
#     pairs: pd.DataFrame,
#     path_generator: PathGenerator,
#     path_embedding_strategy: PathEmbeddingStrategy,
#     category_encoder: OneHotEncoder,
#     relation_encoder: OneHotEncoder,
#     include_directions: bool = False,
# ) -> Dict[str, KGPaths]:
#     """Make MOA predictions on the pairs dataset for reporting purposes.

#     Args:
#         model: The model to make predictions with.
#         runner: The Neo4j runner.
#         pairs: Dataset of drug-disease pairs. Expected columns: (drug_id, disease_id).
#         path_generator: Path generator outputting all paths of interest between a given drug-disease pair.
#         path_embedding_strategy: Path embedding strategy.
#         category_encoder: One-hot encoder for node categories.
#         relation_encoder: One-hot encoder for edge relations.
#         include_directions: Whether to include the direction of the paths in the prediction reports.
#     """
#     pass
