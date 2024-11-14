"""Module with nodes for moa extraction."""

import pandas as pd
import numpy as np
import json

from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional

from pyspark.sql import DataFrame

from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator

from refit.v1.core.inject import inject_object

from .path_embeddings import OneHotEncoder, PathEmbeddingStrategy
from .path_mapping import PathMapper
from .path_generators import PathGenerator
from matrix.pipelines.embeddings.nodes import GraphDB
from matrix.datasets.paths import KGPaths
from matrix.pipelines.modelling.nodes import apply_splitter


def _tag_edges_between_types(
    runner: GraphDB,
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
        runner: The GraphDB object representing the KG..
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
            updated = result[0]["count"]  # [0]
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
        RETURN count(r) AS count
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
                RETURN count(r) AS count
            """)


@inject_object()
def add_tags(
    runner: GraphDB,
    drug_types: List[str],
    disease_types: List[str],
    batch_size: int,
    verbose: bool,
    edges: DataFrame,
    prefix: str = "_moa_extraction_",
) -> None:
    """Add tags to the Neo4j database.

    Args:
        runner: The GraphDB object representing the KG.
        drug_types: List of KG node types representing drugs.
        disease_types: List of KG node types representing diseases.
        batch_size: The batch size to use for the query.
        verbose: Whether to print the number of batches completed.
        prefix: The prefix to add to the tag.
        edges: The edges dataframe. Dummy variable ensuring that the node is run after edges have been added to the KG.
    """
    _tag_edges_between_types(runner, drug_types, disease_types, "drug_disease", batch_size, verbose, prefix)
    _tag_edges_between_types(runner, drug_types, drug_types, "drug_drug", batch_size, verbose, prefix)
    _tag_edges_between_types(runner, disease_types, disease_types, "disease_disease", batch_size, verbose, prefix)
    return {"success": True}


def get_one_hot_encodings(
    nodes: DataFrame,
    edges: DataFrame,
) -> Tuple[OneHotEncoder, OneHotEncoder]:
    """Get the one-hot encodings for node categories and edge relations.

    Args:
        nodes: Nodes dataframe.
        edges: Edges dataframe.

    Returns:
        A tuple of OneHotEncoder objects for node categories and edge relations.
    """
    # Get the node categories
    node_categories = nodes.select("category").distinct().collect()
    node_categories = [row.category for row in node_categories]

    # Get the edge relations
    edge_relations = edges.select("predicate").distinct().collect()
    edge_relations = [row.predicate for row in edge_relations]

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
    add_tags_dummy: dict,
) -> KGPaths:
    """Map the DrugMechDB indication paths to 2 and 3-hop paths in the graph.

    Args:
        runner: The GraphDB object representing the KG.
        drug_mech_db: The DrugMechDB indication paths.
        mapper: Strategy for mapping paths to the graph.
        drugmechdb_entities: The normalized DrugMechDB entities.
        add_tags_dummy: Dummy variable ensuring that the node is run after tags have been added to the KG.
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
    report["entities_in_drugmechdb"] = len(drugmechdb_entities)
    report["proportion_entities_mapped"] = (
        sum(~drugmechdb_entities["mapped_ID"].isna()) / len(drugmechdb_entities) if len(drugmechdb_entities) > 0 else 0
    )
    return json.loads(json.dumps(report, default=float))


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
    df_splits = apply_splitter(df, splitter)
    return KGPaths(df=df_splits).df


@inject_object()
def generate_negative_paths(
    paths: KGPaths,
    negative_sampler_list: List[PathGenerator],
    runner: GraphDB,
) -> KGPaths:
    """Enrich a dataset of positive indication paths with negative samples.

    Args:
        paths: Dataset of positive indication paths with splits information.
        negative_sampler_list: List of path generators for negative path samplers.
        runner: The GraphDB object representing the KG..
    """
    # Add synthesised negative paths to the dataset
    for split in ["TRAIN", "TEST"]:
        for negative_sampler in negative_sampler_list:
            paths_split = KGPaths(df=paths.df[paths.df["split"] == split])
            negative_paths = negative_sampler.run(paths_split, runner)
            negative_paths.df["split"] = split
            negative_paths.df["y"] = 0
            paths.df = pd.concat([paths.df, negative_paths.df])

    # Remove duplicate paths keeping maximum y value
    # This ensures that ground truth positives do not appear as synthesised negatives
    ignore_cols = ["split", "y"]
    dedup_cols = [col for col in paths.df.columns if col not in ignore_cols]
    paths.df = (
        paths.df.sort_values("y", ascending=False)
        .drop_duplicates(subset=dedup_cols, keep="first")
        .reset_index(drop=True)
    )

    # Shuffle the paths
    paths.df = paths.df.sample(frac=1).reset_index(drop=True)

    return paths.df


@inject_object()
def train_model(
    tuner: Any,
    paths: KGPaths,
    path_embedding_strategy: PathEmbeddingStrategy,
    category_encoder: OneHotEncoder,
    relation_encoder: OneHotEncoder,
) -> BaseEstimator:
    """Train the model on the entire paths dataset provided.

    Args:
        tuner: Tuner object.
        paths: The paths dataset with a "y" column.
        path_embedding_strategy: Path embedding strategy.
        category_encoder: One-hot encoder for node categories.
        relation_encoder: One-hot encoder for edge relations.
    """
    X = path_embedding_strategy.run(paths, category_encoder, relation_encoder).astype(np.float32)
    y = paths.df["y"].to_numpy().astype(np.float32).reshape(-1, 1)
    model = tuner.fit(X, y)
    model.fit(X, y)
    return model


@inject_object()
def train_model_split(
    tuner: Any,
    paths: KGPaths,
    path_embedding_strategy: PathEmbeddingStrategy,
    category_encoder: OneHotEncoder,
    relation_encoder: OneHotEncoder,
) -> BaseEstimator:
    """Train the model on the training portion of the paths dataset only.

    Args:
        tuner: Tuner object.
        paths: The paths dataset with a "y" column and split information.
        path_embedding_strategy: Path embedding strategy.
        category_encoder: One-hot encoder for node categories.
        relation_encoder: One-hot encoder for edge relations.
    """
    paths_train = KGPaths(df=paths.df[paths.df["split"] == "TRAIN"])
    return train_model(tuner, paths_train, path_embedding_strategy, category_encoder, relation_encoder)


def make_predictions(
    model: BaseEstimator,
    runner: GraphDB,
    pairs: pd.DataFrame,
    path_generator: PathGenerator,
    path_embedding_strategy: PathEmbeddingStrategy,
    category_encoder: OneHotEncoder,
    relation_encoder: OneHotEncoder,
    score_col_name: str = "MOA_score",
) -> Dict[str, KGPaths]:
    """Make MOA predictions on the pairs dataset.

    Args:
        model: The model to make predictions with.
        pairs: Dataset of drug-disease pairs. Expected columns: (source_id, target_id).
        path_generator: Path generator outputting all paths of interest between a given drug disease pair.
        path_embedding_strategy: Path embedding strategy.
        category_encoder: One-hot encoder for node categories.
        relation_encoder: One-hot encoder for edge relations.
        score_col_name: The name of the column to use for the confidence score.
    Returns:
        Dictionary of KGPaths objects, one for each pair. Each object contains a dataframe with the paths sorted by confidence score.
    """
    paths_dict = dict()
    for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
        # Generate all paths for the pair
        all_paths_for_pair = path_generator.run(runner, row.source_id, row.target_id)

        if len(all_paths_for_pair.df) == 0:
            paths_dict["__".join((row.source_id, row.target_id))] = all_paths_for_pair
            continue

        # Embed the paths
        X = path_embedding_strategy.run(all_paths_for_pair, category_encoder, relation_encoder)
        X = X.astype(np.float32)

        # Make predictions
        y_pred = model.predict_proba(X)[:, 1]  # Returns column vector
        y_pred = y_pred.reshape(-1, 1)

        # Add predictions to the paths dataframe and sort
        all_paths_for_pair.df[score_col_name] = y_pred
        all_paths_for_pair.df = all_paths_for_pair.df.sort_values(by=score_col_name, ascending=False)
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
    runner: GraphDB,
    positive_paths: KGPaths,
    path_generator: PathGenerator,
    path_embedding_strategy: PathEmbeddingStrategy,
    category_encoder: OneHotEncoder,
    relation_encoder: OneHotEncoder,
) -> Dict[str, KGPaths]:
    """Make MOA predictions on the pairs dataset.

    Args:
        model: The model to make predictions with.
        runner: The GraphDB object representing the KG..
        positive_paths: Dataset of positive indication paths, with a "split" information.
        path_generator: Path generator outputting all paths of interest between a given drug-disease pair.
        path_embedding_strategy: Path embedding strategy.
        category_encoder: One-hot encoder for node categories.
        relation_encoder: One-hot encoder for edge relations.

    Returns:
        Dictionary of KGPaths objects, one for each pair.
    """
    test_pairs = _give_test_pairs(positive_paths)
    return make_predictions(
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

    return json.loads(json.dumps(report, default=float))


@inject_object()
def make_output_predictions(
    model: BaseEstimator,
    runner: GraphDB,
    pairs: pd.DataFrame,
    path_generator: PathGenerator,
    path_embedding_strategy: PathEmbeddingStrategy,
    category_encoder: OneHotEncoder,
    relation_encoder: OneHotEncoder,
    metrics_dummy: Any,  # TODO: Remove or add to docstring
    drug_col_name: str = "source_id",
    disease_col_name: str = "target_id",
    num_pairs_limit: Optional[int] = None,
    score_col_name: str = "MOA_score",
) -> Dict[str, KGPaths]:
    """Make MOA predictions on the pairs dataset.

    Args:
        model: The model to make predictions with.
        runner: The GraphDB object representing the KG..
        pairs: Dataset of drug-disease pairs. Expected columns: (source_id, target_id).
        path_generator: Path generator outputting all paths of interest between a given drug-disease pair.
        path_embedding_strategy: Path embedding strategy.
        category_encoder: One-hot encoder for node categories.
        relation_encoder: One-hot encoder for edge relations.
        drug_col_name: The name of the column containing the drug IDs.
        disease_col_name: The name of the column containing the disease IDs.
        num_pairs_limit: Optional cut-off for the number of pairs. For testing purposes. Defaults to None.
        score_col_name: The name of the column to use for the confidence score.

    Returns:
        Dictionary of KGPaths objects, one for each pair.
    """
    if type(num_pairs_limit) is str:
        num_pairs_limit = eval(num_pairs_limit)  # eval() to allow None type injection
    pairs = pairs.head(num_pairs_limit) if num_pairs_limit is not None else pairs
    pairs = pairs.rename(columns={drug_col_name: "source_id", disease_col_name: "target_id"})

    return make_predictions(
        model=model,
        runner=runner,
        pairs=pairs,
        path_generator=path_generator,
        path_embedding_strategy=path_embedding_strategy,
        category_encoder=category_encoder,
        relation_encoder=relation_encoder,
        score_col_name=score_col_name,
    )


def _squash_directionality(paths: KGPaths, suffix: str = "*", score_col_name: str = "MOA_score") -> pd.DataFrame:
    """Squash the directionality of the paths.

    Predicates corresponding to backwards edges are marked by a suffix.

    Args:
        paths: KGPaths object containing the paths with a confidence score.
        suffix: The suffix to append to the predicates corresponding to backwards edges.
        score_col_name: The name of the column to use for the confidence score.

    Returns:
        DataFrame representing the paths with the directionality of the paths squashed.
    """
    # Separate the edge and node columns
    predicate_cols = [f"predicates_{i}" for i in range(1, paths.num_hops + 1)]
    forward_cols = [f"is_forward_{i}" for i in range(1, paths.num_hops + 1)]
    not_node_cols = predicate_cols + forward_cols + [score_col_name]
    node_cols = [col for col in paths.df.columns if col not in not_node_cols]

    # Append suffix to backward predicates and remove the is_forward column
    for is_forward_col, predicate_col in zip(forward_cols, predicate_cols):
        is_backward = paths.df[is_forward_col].eq(False)
        paths.df.loc[is_backward, predicate_col] = paths.df.loc[is_backward, predicate_col] + suffix
    paths.df = paths.df.drop(columns=forward_cols)

    # Squash the directionality
    grouped = paths.df.groupby(node_cols, as_index=False)
    agg_dict = {col: lambda x: ",".join(x.unique()) for col in predicate_cols}
    agg_dict[score_col_name] = "max"
    return grouped.agg(agg_dict).reset_index(drop=True)


def generate_predictions_reports(
    predictions: Dict[str, KGPaths],
    include_edge_directions: bool = True,
    num_paths_per_pair_limit: int = None,
    score_col_name: str = "MOA_score",
) -> Dict[str, pd.DataFrame]:
    """Generates reports for MOA predictions.

    Args:
        predictions: Dictionary of KGPaths objects, one for each pair.
        include_edge_directions: Whether to include the edge directions in the report.
        num_paths_per_pair_limit: Optional cut-off for the number of paths per pair.

    Returns:
        Dictionary of KGPaths objects, one for each pair.
    """
    # Dictionary of Excel report
    reports = dict()
    # Combined dataframes for SQL insertion
    pair_info_dfs = []
    moa_predictions_dfs = []

    for pair_name, predictions_load_func in predictions.items():
        # Load the predictions
        predictions = predictions_load_func()
        N_paths = len(predictions)
        if N_paths == 0:
            reports[pair_name + "_MOA_predictions.xlsx"] = {
                "MOA predictions": pd.DataFrame({"NO PATHS": ["No paths found between the given drug and disease"]})
            }
            continue

        # Squash the directionality
        predictions_df = _squash_directionality(predictions)

        # Create the pair information dataframe
        pair_info = pd.DataFrame(
            {
                "Drug ID": [predictions_df.iloc[0]["source_id"]],
                "Drug Name": [predictions_df.iloc[0]["source_name"]],
                "Disease ID": [predictions_df.iloc[0]["target_id"]],
                "Disease Name": [predictions_df.iloc[0]["target_name"]],
                "Total number of paths between pair": [N_paths],
            }
        )

        # Create the MOA predictions dataframe
        predictions_df = (
            predictions_df.head(num_paths_per_pair_limit) if num_paths_per_pair_limit is not None else predictions_df
        )
        cols = predictions_df.columns.to_list()
        cols = [col for col in cols if "source" not in col]
        cols = [col for col in cols if "target" not in col]
        predictions_df = predictions_df[cols]

        # Add the pair information and MOA predictions to a multiframe to be exported as Excel
        pair_info["Number of displayed paths"] = [len(predictions_df)]
        reports[pair_name + "_MOA_predictions.xlsx"] = {
            "MOA predictions": predictions_df,
            "Pair information": pair_info,
        }

        # Add to combined dataframes with additional pair ID column
        pair_ID = "|".join([pair_info.iloc[0]["Drug ID"], pair_info.iloc[0]["Disease ID"]])
        pair_info["pair_id"] = pair_ID
        predictions_df["pair_id"] = pair_ID
        pair_info_dfs.append(pair_info)
        moa_predictions_dfs.append(predictions_df)

    # Combine all dataframes
    combined_pair_info = pd.concat(pair_info_dfs, ignore_index=True)
    combined_predictions = pd.concat(moa_predictions_dfs, ignore_index=True)

    return {"excel_reports": reports, "pair_info_dfs": combined_pair_info, "moa_predictions_dfs": combined_predictions}
