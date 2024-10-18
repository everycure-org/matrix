"""Module with nodes for matrix generation."""

import logging
from tqdm import tqdm
from typing import List, Dict, Union, Tuple

from sklearn.impute._base import _BaseImputer

import pandas as pd

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.make_list_regexable import _extract_elements_in_list

from matrix.datasets.graph import KnowledgeGraph

from matrix.pipelines.modelling.nodes import apply_transformers
from matrix.pipelines.modelling.model import ModelWrapper

from datetime import datetime

logger = logging.getLogger(__name__)


def enrich_embeddings(
    nodes: DataFrame,
    drugs: DataFrame,
    diseases: DataFrame,
) -> DataFrame:
    """Function to enrich drug and disease list with embeddings.

    Args:
        nodes: Dataframe with node embeddings
        drugs: List of drugs
        diseases: List of diseases
    """
    return (
        drugs.withColumn("is_drug", F.lit(True))
        .unionByName(diseases.withColumn("is_disease", F.lit(True)), allowMissingColumns=True)
        .withColumnRenamed("curie", "id")
        .join(nodes, on="id", how="inner")
        .select("is_drug", "is_disease", "id", "topological_embedding")
        .withColumn("is_drug", F.coalesce(F.col("is_drug"), F.lit(False)))
        .withColumn("is_disease", F.coalesce(F.col("is_disease"), F.lit(False)))
    )


def spark_to_pd(nodes: DataFrame) -> pd.DataFrame:
    """Temporary function to transform spark parquet to pandas parquet.

    Related to https://github.com/everycure-org/matrix/issues/71.
    TODO: replace/remove the function once pyarrow error is fixed.

    Args:
        nodes: Dataframe with node embeddings
    """
    return nodes.toPandas()


@has_schema(
    schema={
        "source": "object",
        "target": "object",
    },
    allow_subset=True,
)
@inject_object()
def generate_pairs(
    drugs: pd.DataFrame,
    diseases: pd.DataFrame,
    graph: KnowledgeGraph,
    known_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Function to generate matrix dataset.

    Args:
        drugs: Dataframe containing IDs for the list of drugs.
        diseases: Dataframe containing IDs for the list of diseases.
        graph: Object containing node embeddings.
        known_pairs: Labelled ground truth drug-disease pairs dataset.

    Returns:
        Pairs dataframe containing all combinations of drugs and diseases that do not lie in the training set.
    """
    # Collect list of drugs and diseases
    drugs_lst = drugs["curie"].tolist()
    diseases_lst = diseases["curie"].tolist()

    # Remove duplicates
    drugs_lst = list(set(drugs_lst))
    diseases_lst = list(set(diseases_lst))

    # Remove drugs and diseases without embeddings
    nodes_with_embeddings = set(graph._nodes["id"])
    drugs_lst = [drug for drug in drugs_lst if drug in nodes_with_embeddings]
    diseases_lst = [disease for disease in diseases_lst if disease in nodes_with_embeddings]

    # Generate all combinations
    matrix_slices = []
    for disease in tqdm(diseases_lst):
        matrix_slice = pd.DataFrame({"source": drugs_lst, "target": disease})
        matrix_slices.append(matrix_slice)

    # Concatenate all slices at once
    matrix = pd.concat(matrix_slices, ignore_index=True)

    # Remove training set and return
    train_pairs = known_pairs[~known_pairs["split"].eq("TEST")]
    train_pairs_set = set(zip(train_pairs["source"], train_pairs["target"]))
    is_in_train = matrix.apply(lambda row: (row["source"], row["target"]) in train_pairs_set, axis=1)
    return matrix[~is_in_train]


def make_batch_predictions(
    graph: KnowledgeGraph,
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    score_col_name: str,
    batch_by: str = "target",
) -> pd.DataFrame:
    """Generate probability scores for drug-disease dataset.

    This function computes the scores in batches to avoid memory issues.

    FUTURE: Experiment with PySpark for predictions where model is broadcasted.
    https://dataking.hashnode.dev/making-predictions-on-a-pyspark-dataframe-with-a-scikit-learn-model-ckzzyrudn01lv25nv41i2ajjh

    Args:
        graph: Knowledge graph.
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        score_col_name: Probability score column name.
        batch_by: Column to use for batching (e.g., "target" or "source").

    Returns:
        Pairs dataset with additional column containing the probability scores.
    """

    def process_batch(batch: pd.DataFrame) -> pd.DataFrame:
        # Collect embedding vectors
        batch["source_embedding"] = batch.apply(lambda row: graph.get_embedding(row.source, default=pd.NA), axis=1)
        batch["target_embedding"] = batch.apply(lambda row: graph.get_embedding(row.target, default=pd.NA), axis=1)

        # Retrieve rows with null embeddings
        # NOTE: This only happens in a rare scenario where the node synonymizer
        # provided an identifier for a node that does _not_ exist in our KG.
        # https://github.com/everycure-org/matrix/issues/409
        removed = batch[batch["source_embedding"].isna() | batch["target_embedding"].isna()]
        if len(removed.index) > 0:
            logger.warning(f"Dropped {len(removed.index)} pairs during generation!")
            logger.warning(
                "Dropped: %s",
                ",".join([f"({r.source}, {r.target})" for _, r in removed.iterrows()]),
            )

        # Drop rows without source/target embeddings
        batch = batch.dropna(subset=["source_embedding", "target_embedding"])

        # Return empty dataframe if all rows are dropped
        if len(batch) == 0:
            return batch.drop(columns=["source_embedding", "target_embedding"])

        # Apply transformers to data
        transformed = apply_transformers(batch, transformers)

        # Extract features
        batch_features = _extract_elements_in_list(transformed.columns, features, raise_exc=True)

        # Generate model probability scores
        batch[score_col_name] = model.predict_proba(transformed[batch_features].values)[:, 1]

        # Drop embedding columns
        batch = batch.drop(columns=["source_embedding", "target_embedding"])
        return batch

    # Group data by the specified prefix
    grouped = data.groupby(batch_by)

    # Process data in batches
    result_parts = []
    for _, batch in tqdm(grouped):
        result_parts.append(process_batch(batch))

    # Combine results
    results = pd.concat(result_parts, axis=0)

    return results


def make_predictions_and_sort(
    graph: KnowledgeGraph,
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    score_col_name: str,
    batch_by: str,
) -> pd.DataFrame:
    """Generate and sort probability scores for a drug-disease dataset.

    FUTURE: Perform parallelised computation instead of batching with a for loop.

    Args:
        graph: Knowledge graph.
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        score_col_name: Probability score column name.
        batch_by: Column to use for batching (e.g., "target" or "source").

    Returns:
        Pairs dataset sorted by an additional column containing the probability scores.
    """
    # Generate scores
    data = make_batch_predictions(graph, data, transformers, model, features, score_col_name, batch_by=batch_by)

    # Sort by the probability score
    sorted_data = data.sort_values(by=score_col_name, ascending=False)
    return sorted_data


def generate_summary_metadata(meta_col_names, score_col_name, stats_col_names):
    """
    Generate metadata for the output matrix.

    Args:
        meta_col_names (dict): Dictionary containing metadata column names.
        score_col_name (str): Name of the score column.
        stats_col_names (dict): Dictionary containing statistical column names.

    Returns:
        pd.DataFrame: DataFrame containing summary metadata.
    """
    summary_metadata = {}

    # Add metadata for ID columns
    summary_metadata.update(meta_col_names["drug_list"])
    summary_metadata.update(meta_col_names["disease_list"])

    # Add metadata for score columns
    summary_metadata[score_col_name] = "Probability score"
    summary_metadata["pair_id"] = "Unique identifier for each pair"
    summary_metadata["is_known_positive"] = "Whether the pair is a known positive"
    summary_metadata["is_known_negative"] = "Whether the pair is a known negative"

    # Add metadata for KG columns
    summary_metadata.update(meta_col_names["kg_data"])

    # Add metadata for statistical columns
    for stat, description in stats_col_names["per_disease"]["top"].items():
        summary_metadata[f"{stat}_top_per_disease"] = f"{description} in the top n_reporting pairs"
    for stat, description in stats_col_names["per_disease"]["all"].items():
        summary_metadata[f"{stat}_all_per_disease"] = f"{description} in all pairs"

    return pd.DataFrame(list(summary_metadata.items()), columns=["Column", "Explanation"])


def process_top_pairs(
    data: pd.DataFrame, n_reporting: int, drugs: pd.DataFrame, diseases: pd.DataFrame, score_col_name: str
) -> pd.DataFrame:
    """
    Process the top pairs from the data and add additional information.

    Args:
        data (pd.DataFrame): The input DataFrame containing all pairs.
        n_reporting (int): The number of top pairs to process.
        drugs (pd.DataFrame): DataFrame containing drug information.
        diseases (pd.DataFrame): DataFrame containing disease information.
        score_col_name (str): The name of the column containing the score.

    Returns:
        pd.DataFrame: Processed DataFrame containing the top pairs with additional information.
    """
    top_pairs = data.head(n_reporting).copy()

    # Generate mapping dictionaries
    drug_mappings = {
        "kg_name": {row["curie"]: row["name"] for _, row in drugs.iterrows()},
        "list_id": {row["curie"]: row["single_ID"] for _, row in drugs.iterrows()},
        "list_name": {row["curie"]: row["ID_Label"] for _, row in drugs.iterrows()},
    }

    disease_mappings = {
        "kg_name": {row["curie"]: row["name"] for _, row in diseases.iterrows()},
        "list_id": {row["curie"]: row["category_class"] for _, row in diseases.iterrows()},
        "list_name": {row["curie"]: row["label"] for _, row in diseases.iterrows()},
    }

    # Add additional information
    top_pairs["kg_drug_name"] = top_pairs["source"].map(drug_mappings["kg_name"])
    top_pairs["kg_disease_name"] = top_pairs["target"].map(disease_mappings["kg_name"])
    top_pairs["disease_id"] = top_pairs["target"].map(disease_mappings["list_id"])
    top_pairs["disease_name"] = top_pairs["target"].map(disease_mappings["list_name"])
    top_pairs["drug_id"] = top_pairs["source"].map(drug_mappings["list_id"])
    top_pairs["drug_name"] = top_pairs["source"].map(drug_mappings["list_name"])

    # Rename ID columns and add pair ID
    top_pairs = top_pairs.rename(columns={"source": "kg_drug_id", "target": "kg_disease_id"})
    top_pairs["pair_id"] = top_pairs["drug_id"] + "|" + top_pairs["disease_id"]

    return top_pairs


def add_descriptive_stats(
    top_pairs: pd.DataFrame, data: pd.DataFrame, stats_col_names: Dict, score_col_name: str
) -> pd.DataFrame:
    """
    Add descriptive statistics to the top pairs DataFrame.

    Args:
        top_pairs (pd.DataFrame): DataFrame containing the top pairs.
        data (pd.DataFrame): The full dataset containing all pairs.
        stats_col_names (Dict): Dictionary containing the names of statistical columns.
        score_col_name (str): The name of the column containing the score.

    Returns:
        pd.DataFrame: DataFrame with added descriptive statistics.
    """
    for stat in stats_col_names["per_disease"]["top"].keys():
        top_pairs[f"{stat}_top_per_disease"] = top_pairs.groupby("kg_disease_id")[score_col_name].transform(stat)

    top_pairs_all = data[
        data["target"].isin(top_pairs["kg_disease_id"].unique()) | data["source"].isin(top_pairs["kg_drug_id"].unique())
    ]

    for stat in stats_col_names["per_disease"]["all"].keys():
        stat_dict = top_pairs_all.groupby("target")[score_col_name].agg(stat).to_dict()
        top_pairs[f"{stat}_all_per_disease"] = top_pairs["kg_disease_id"].map(stat_dict)

    return top_pairs


def flag_known_pairs(top_pairs: pd.DataFrame, known_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Flag known positive and negative pairs in the top pairs DataFrame.

    Args:
        top_pairs (pd.DataFrame): DataFrame containing the top pairs.
        known_pairs (pd.DataFrame): DataFrame containing known positive and negative pairs.

    Returns:
        pd.DataFrame: DataFrame with added flags for known positive and negative pairs.
    """
    known_pos_pairs_set = set(
        zip(known_pairs[known_pairs["y"].eq(1)]["source"], known_pairs[known_pairs["y"].eq(1)]["target"])
    )
    known_neg_pairs_set = set(
        zip(known_pairs[~known_pairs["y"].eq(1)]["source"], known_pairs[~known_pairs["y"].eq(1)]["target"])
    )

    top_pairs["is_known_positive"] = top_pairs.apply(
        lambda row: (row["kg_drug_id"], row["kg_disease_id"]) in known_pos_pairs_set, axis=1
    )
    top_pairs["is_known_negative"] = top_pairs.apply(
        lambda row: (row["kg_drug_id"], row["kg_disease_id"]) in known_neg_pairs_set, axis=1
    )

    return top_pairs


def reorder_columns(top_pairs: pd.DataFrame, score_col_name: str, matrix_params: Dict) -> pd.DataFrame:
    """
    Reorder columns in the top pairs DataFrame.

    Args:
        top_pairs (pd.DataFrame): DataFrame containing the top pairs.
        score_col_name (str): The name of the column containing the score.
        matrix_params (Dict): Dictionary containing matrix parameters.

    Returns:
        pd.DataFrame: DataFrame with reordered columns.
    """
    meta_col_names = matrix_params["metadata"]
    stats_col_names = matrix_params["stats_col_names"]
    tags = matrix_params["tags"]

    id_columns = list(meta_col_names["drug_list"].keys()) + list(meta_col_names["disease_list"].keys())
    score_columns = [score_col_name]
    tag_columns = list(tags["drugs"].keys()) + list(tags["diseases"].keys())
    kg_columns = list(meta_col_names["kg_data"].keys())
    stat_columns = list(stats_col_names["per_disease"]["top"].keys())
    stat_suffixes = ["_top_per_disease", "_all_per_disease"]
    columns_order = (
        id_columns
        + score_columns
        + kg_columns
        + tag_columns
        + [f"{stat}{suffix}" for stat in stat_columns for suffix in stat_suffixes]
    )

    # Remove columns that are not in the top pairs DataFrame but are specified in params
    columns_order = [col for col in columns_order if col in top_pairs.columns]
    return top_pairs[columns_order]


def add_tags(top_pairs: pd.DataFrame, drugs: pd.DataFrame, diseases: pd.DataFrame, matrix_params: Dict) -> pd.DataFrame:
    """
    Add tag columns to the top pairs DataFrame.

    Args:
        top_pairs (pd.DataFrame): DataFrame containing the top pairs.
        drugs (pd.DataFrame): DataFrame containing drug information.
        diseases (pd.DataFrame): DataFrame containing disease information.
        matrix_params (Dict): Dictionary containing matrix parameters.

    Returns:
        pd.DataFrame: DataFrame with added tag columns.
    """
    # Add tag columns for drugs and diseasesto the top pairs DataFrame
    for set, set_id, df in [("drugs", "drug_id", drugs), ("diseases", "disease_id", diseases)]:
        for tag_name, _ in matrix_params.get(set, {}).items():
            if tag_name not in df.columns:
                logger.warning(f"Tag column '{tag_name}' not found in {set} DataFrame. Skipping.")
            else:
                tag_mapping = dict(zip(df["curie"], df[tag_name]))

                # Add the tag to top_pairs
                top_pairs[tag_name] = top_pairs[set_id].map(tag_mapping)
    return top_pairs


def generate_report(
    data: pd.DataFrame,
    n_reporting: int,
    drugs: pd.DataFrame,
    diseases: pd.DataFrame,
    known_pairs: pd.DataFrame,
    score_col_name: str,
    matrix_params: Dict,
) -> List[pd.DataFrame]:
    """Generates a report with the top pairs and metadata.

    Args:
        data: Pairs dataset.
        n_reporting: Number of pairs in the report.
        drugs: Dataframe containing names and IDs for the list of drugs.
        diseases: Dataframe containing names and IDs for the list of diseases.
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        matrix_params: Dictionary containing matrix metadata and other meters.
    Returns:
        Dataframe with the top pairs and additional information for the drugs and diseases.
    """
    stats = matrix_params.get("stats_col_names")
    tags = matrix_params.get("tags")
    top_pairs = process_top_pairs(data, n_reporting, drugs, diseases, score_col_name)
    top_pairs = add_descriptive_stats(top_pairs, data, stats, score_col_name)
    top_pairs = flag_known_pairs(top_pairs, known_pairs)
    top_pairs = add_tags(top_pairs, drugs, diseases, tags)
    top_pairs = reorder_columns(top_pairs, score_col_name, matrix_params)
    return top_pairs


def generate_metadata(
    matrix_report: pd.DataFrame,
    score_col_name: str,
    matrix_params: Dict,
    run_metadata: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a metadata report.

    Args:
        matrix_report: pd.DataFrame, dummy variable to maintain proper lineage to be logged within metadata
        score_col_name: Probability score column name.
        matrix_params: Dictionary of column names and their descriptions.
        run_metadata: Dictionary of run metadata.
    Returns:
        Tuple containing:
        - Dataframe containing metadata such as data sources version, timestamp, run name etc.
        - Dataframe with metadata about the output matrix columns.
    """
    dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for key, value in run_metadata.items():
        dict[key] = value
    metadata = matrix_params.get("metadata")
    stats = matrix_params.get("stats_col_names")
    summary_metadata = generate_summary_metadata(metadata, score_col_name, stats)
    return pd.DataFrame(list(run_metadata.items()), columns=["Key", "Value"]), summary_metadata
