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


def _add_flag_columns(matrix: pd.DataFrame, known_pairs: pd.DataFrame, clinical_trials: pd.DataFrame) -> pd.DataFrame:
    """Adds boolean columns flagging known positives and known negatives.

    Args:
        matrix: Drug-disease pairs dataset.
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        clinical_trials: Pairs dataset representing outcomes of recent clinical trials.

    Returns:
        Pairs dataset with flag columns.
    """

    def create_flag_column(pairs):
        pairs_set = set(zip(pairs["source"], pairs["target"]))
        return matrix.apply(lambda row: (row["source"], row["target"]) in pairs_set, axis=1)

    # Flag known positives and negatives
    test_pairs = known_pairs[known_pairs["split"].eq("TEST")]
    test_pair_is_pos = test_pairs["y"].eq(1)
    test_pos_pairs = test_pairs[test_pair_is_pos]
    test_neg_pairs = test_pairs[~test_pair_is_pos]
    matrix["is_known_positive"] = create_flag_column(test_pos_pairs)
    matrix["is_known_negative"] = create_flag_column(test_neg_pairs)

    # Flag clinical trials data
    clinical_trials = clinical_trials.rename(columns={"drug_kg_curie": "source", "disease_kg_curie": "target"})
    matrix["trial_sig_better"] = create_flag_column(clinical_trials[clinical_trials["significantly_better"] == 1])
    matrix["trial_non_sig_better"] = create_flag_column(
        clinical_trials[clinical_trials["non_significantly_better"] == 1]
    )
    matrix["trial_sig_worse"] = create_flag_column(clinical_trials[clinical_trials["non_significantly_worse"] == 1])
    matrix["trial_non_sig_worse"] = create_flag_column(clinical_trials[clinical_trials["significantly_worse"] == 1])

    return matrix


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
        "is_known_positive": "bool",
        "is_known_negative": "bool",
        "trial_sig_better": "bool",
        "trial_non_sig_better": "bool",
        "trial_sig_worse": "bool",
        "trial_non_sig_worse": "bool",
    },
    allow_subset=True,
)
@inject_object()
def generate_pairs(
    drugs: pd.DataFrame,
    diseases: pd.DataFrame,
    graph: KnowledgeGraph,
    known_pairs: pd.DataFrame,
    clinical_trials: pd.DataFrame,
) -> pd.DataFrame:
    """Function to generate matrix dataset.

    FUTURE: Consider rewriting operations in PySpark for speed

    Args:
        drugs: Dataframe containing IDs for the list of drugs.
        diseases: Dataframe containing IDs for the list of diseases.
        graph: Object containing node embeddings.
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        clinical_trials: Pairs dataset representing outcomes of recent clinical trials.

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

    # Remove training set
    train_pairs = known_pairs[~known_pairs["split"].eq("TEST")]
    train_pairs_set = set(zip(train_pairs["source"], train_pairs["target"]))
    is_in_train = matrix.apply(lambda row: (row["source"], row["target"]) in train_pairs_set, axis=1)
    matrix = matrix[~is_in_train]

    # Add flag columns for known positives and negatives
    matrix = _add_flag_columns(matrix, known_pairs, clinical_trials)

    return matrix


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


def generate_summary_metadata(matrix_parameters: Dict, score_col_name: str) -> pd.DataFrame:
    """
    Generate metadata for the output matrix.

    Args:
        matrix_parameters (Dict): Dictionary containing matrix parameters.
        score_col_name (str): Name of the score column.

    Returns:
        pd.DataFrame: DataFrame containing summary metadata.
    """
    summary_metadata = {}

    meta_col_names = matrix_parameters["metadata"]
    stats_col_names = matrix_parameters["stats_col_names"]
    tags_col_names = matrix_parameters["tags"]

    # Add metadata for ID columns
    summary_metadata.update(meta_col_names["drug_list"])
    summary_metadata.update(meta_col_names["disease_list"])

    # Add metadata for KG columns and tags
    summary_metadata.update(meta_col_names["kg_data"])

    # Add metadata for tags and filters
    summary_metadata.update(tags_col_names["drugs"])
    summary_metadata.update(tags_col_names["pairs"])
    summary_metadata.update(tags_col_names["diseases"])

    # Add metadata for statistical columns
    for stat, description in stats_col_names["per_disease"]["top"].items():
        summary_metadata[f"{stat}_top_per_disease"] = f"{description} in the top n_reporting pairs"
    for stat, description in stats_col_names["per_disease"]["all"].items():
        summary_metadata[f"{stat}_all_per_disease"] = f"{description} in all pairs"

    return pd.DataFrame(list(summary_metadata.items()), columns=["Key", "Value"])


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
    data = data.sort_values(by=score_col_name, ascending=False)
    for entity_type, df_col, id_col in [("disease", "target", "kg_disease_id"), ("drug", "source", "kg_drug_id")]:
        # Calculate stats for top pairs
        for stat in stats_col_names[f"per_{entity_type}"]["top"].keys():
            top_pairs[f"{stat}_top_per_{entity_type}"] = top_pairs.groupby(id_col)[score_col_name].transform(stat)

        # Calculate stats for all pairs (need to use different df)
        all_pairs = data[data[df_col].isin(top_pairs[id_col].unique())]
        for stat in stats_col_names[f"per_{entity_type}"]["all"].keys():
            stat_dict = all_pairs.groupby(df_col)[score_col_name].agg(stat)
            top_pairs[f"{stat}_all_per_{entity_type}"] = top_pairs[id_col].map(stat_dict)
    return top_pairs


def flag_known_pairs(top_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Flag known positive and negative pairs in the top pairs DataFrame.

    Args:
        top_pairs (pd.DataFrame): DataFrame containing the top pairs.

    Returns:
        pd.DataFrame: DataFrame with added flags for known positive and negative pairs.
    """
    top_pairs["is_known_positive"] = (
        top_pairs["is_known_positive"] | top_pairs["trial_sig_better"] | top_pairs["trial_non_sig_better"]
    )
    top_pairs["is_known_negative"] = (
        top_pairs["is_known_negative"] | top_pairs["trial_sig_worse"] | top_pairs["trial_non_sig_worse"]
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
    tag_columns = list(tags["drugs"].keys()) + list(tags["diseases"].keys()) + list(tags["pairs"].keys())
    kg_columns = list(meta_col_names["kg_data"].keys())
    stat_columns = list()
    for main_key in ["per_disease", "per_drug"]:
        for sub_key in ["top", "all"]:
            stat_columns = stat_columns + [
                f"{stat}_{sub_key}_{main_key}" for stat in list(stats_col_names[main_key][sub_key].keys())
            ]
    columns_order = id_columns + score_columns + kg_columns + tag_columns + stat_columns

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
    top_pairs = flag_known_pairs(top_pairs)
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
    meta_dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for key, value in run_metadata.items():
        if key == "versions":
            for subkey, subvalue in value.items():
                meta_dict[f"{subkey}_version"] = subvalue["version"]
        else:
            meta_dict[key] = value

    # Generate legends column and filter out based on
    legends_df = generate_summary_metadata(matrix_params, score_col_name)
    legends_df = legends_df.loc[legends_df["Key"].isin(matrix_report.columns.values)]
    # Generate metadata df
    version_df = pd.DataFrame(list(meta_dict.items()), columns=["Key", "Value"])
    # Concatenate version and legends dfs
    return pd.concat([version_df, legends_df], axis=0)
