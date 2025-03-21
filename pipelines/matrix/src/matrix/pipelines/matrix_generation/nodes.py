import functools
import itertools
import logging
import operator
import os
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
from matrix.datasets.graph import KnowledgeGraph
from matrix.inject import _extract_elements_in_list, inject_object
from matrix.pipelines.modelling.model import ModelWrapper
from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output
from pyspark.sql import Row, Window
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType
from sklearn.impute._base import _BaseImputer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def enrich_embeddings(
    nodes: ps.DataFrame,
    drugs: ps.DataFrame,
    diseases: ps.DataFrame,
) -> ps.DataFrame:
    """Function to enrich drug and disease list with embeddings.

    Args:
        nodes: Dataframe with node embeddings
        drugs: List of drugs
        diseases: List of diseases
    """
    return (
        drugs.withColumn("is_drug", F.lit(True))
        .unionByName(diseases.withColumn("is_disease", F.lit(True)), allowMissingColumns=True)
        .join(nodes, on="id", how="inner")
        .select("is_drug", "is_disease", "id", "topological_embedding")
        .withColumn("is_drug", F.coalesce(F.col("is_drug"), F.lit(False)))
        .withColumn("is_disease", F.coalesce(F.col("is_disease"), F.lit(False)))
    )


def _add_flag_columns(
    matrix: pd.DataFrame, known_pairs: pd.DataFrame, clinical_trials: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
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
        # Ensure the function returns a Series
        result = matrix.apply(lambda row: (row["source"], row["target"]) in pairs_set, axis=1)

        return result.astype(bool)

    # Flag known positives and negatives
    test_pairs = known_pairs[known_pairs["split"].eq("TEST")]
    test_pair_is_pos = test_pairs["y"].eq(1)
    test_pos_pairs = test_pairs[test_pair_is_pos]
    test_neg_pairs = test_pairs[~test_pair_is_pos]
    matrix["is_known_positive"] = create_flag_column(test_pos_pairs)
    matrix["is_known_negative"] = create_flag_column(test_neg_pairs)

    # TODO: Need to make this dynamic
    # Flag clinical trials data
    clinical_trials = clinical_trials.rename(columns={"subject": "source", "object": "target"})
    matrix["trial_sig_better"] = create_flag_column(clinical_trials[clinical_trials["significantly_better"] == 1])
    matrix["trial_non_sig_better"] = create_flag_column(
        clinical_trials[clinical_trials["non_significantly_better"] == 1]
    )
    matrix["trial_sig_worse"] = create_flag_column(clinical_trials[clinical_trials["non_significantly_worse"] == 1])
    matrix["trial_non_sig_worse"] = create_flag_column(clinical_trials[clinical_trials["significantly_worse"] == 1])

    return matrix


@check_output(
    schema=DataFrameSchema(
        columns={
            "source": Column(str, nullable=False),
            "target": Column(str, nullable=False),
            "is_known_positive": Column(bool, nullable=False),
            "is_known_negative": Column(bool, nullable=False),
            "trial_sig_better": Column(bool, nullable=False),
            "trial_non_sig_better": Column(bool, nullable=False),
            "trial_sig_worse": Column(bool, nullable=False),
            "trial_non_sig_worse": Column(bool, nullable=False),
        },
        unique=["source", "target"],
    )
)
@inject_object()
def generate_pairs(
    known_pairs: pd.DataFrame,
    drugs: pd.DataFrame,
    diseases: pd.DataFrame,
    graph: KnowledgeGraph,
    clinical_trials: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Function to generate matrix dataset.

    FUTURE: Consider rewriting operations in PySpark for speed

    Args:
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        drugs: Dataframe containing IDs for the list of drugs.
        diseases: Dataframe containing IDs for the list of diseases.
        graph: Object containing node embeddings.
        clinical_trials: Pairs dataset representing outcomes of recent clinical trials.

    Returns:
        Pairs dataframe containing all combinations of drugs and diseases that do not lie in the training set.
    """
    # Collect list of drugs and diseases
    drugs_lst = drugs["id"].tolist()
    diseases_lst = diseases["id"].tolist()

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


def make_predictions_and_sort(
    graph: KnowledgeGraph,
    data: ps.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    treat_score_col_name: str,
    not_treat_score_col_name: str,
    unknown_score_col_name: str,
) -> ps.DataFrame:
    """Generate probability scores for drug-disease dataset.

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

    # 1. convert the knowledgeGraph into a Spark DataFrame.
    #    Use `data.sparkSession.createDataFrame` so you immediately have the SparkSession you need (and don't need to call it).
    # 2. Replace graph.get_embedding with a simple Spark join.
    if "ARGO_NODE_ID" not in os.environ:
        data = data.limit(1000)
    schema = StructType(
        [StructField("id", StringType(), True), StructField("topological_embedding", ArrayType(FloatType()), True)]
    )

    embeddings = data.sparkSession.createDataFrame(
        [(k, v.tolist()) for k, v in graph._embeddings.items()], schema=schema
    ).cache()
    logger.info(f"rows in embeddings lookup table: {embeddings.count()}")

    data = data.join(
        embeddings.withColumnsRenamed({"id": "source", "topological_embedding": "source_embedding"}),
        on="source",
        how="left",
    ).join(
        embeddings.withColumnsRenamed({"id": "target", "topological_embedding": "target_embedding"}),
        on="target",
        how="left",
    )
    logger.info(f"data size: {data.count():_}x{len(data.columns):_}")
    # Retrieve rows with null embeddings
    # NOTE: This only happens in a rare scenario where the node synonymizer
    # provided an identifier for a node that does _not_ exist in our KG.
    # https://github.com/everycure-org/matrix/issues/409

    features: list[str] = list(itertools.chain.from_iterable(x["features"] for x in transformers.values()))
    print(f"features: {features}")
    if logger.isEnabledFor(logging.INFO):
        logging.info(f"Checking for dropped pairs because one of the features ({features}) is empty…")
        removed = (
            data.filter(functools.reduce(operator.or_, (F.col(colname).isNull() for colname in features)))
            .select("source", "target")
            .cache()
        )
        removed_pairs = removed.take(50)  # Can be extended, but mind OOM.
        if removed_pairs:
            logger.warning(f"Dropped {removed.count()} pairs during generation!")
            logger.warning("Dropped (subset of 50 shown): %s", ", ".join(f"({p[0]}, {p[1]})" for p in removed_pairs))
        removed.unpersist()

    # Drop rows without source/target embeddings
    data = data.dropna(subset=features)

    # # Return empty dataframe if all rows are dropped
    # if data.isEmpty():
    #     return data.drop("source_embedding", "target_embedding")

    # Apply transformers to data (assuming this can work with PySpark)
    # transformed = apply_transformers(data, transformers.values())
    feature_col = "_source_and_target"
    transformed = data.withColumn(feature_col, F.concat(*features)).drop(*features)
    # Extract features
    # data_features = _extract_elements_in_list(transformed.columns, features, True)

    def predict_partition(partitionindex: int, partition: Iterable[Row]) -> Iterator[Row]:
        partition_df = pd.DataFrame.from_records(row.asDict() for row in partition)
        if partition_df.empty:
            logger.warning(f"partition with index {partitionindex} is empty")
            return

        logger.info(f"cols before: {partition_df.columns}")
        s = partition_df.pop(feature_col)
        logger.info(f"cols after: {partition_df.columns}")
        logger.info(s.head(3))
        X = pd.DataFrame.from_dict(dict(zip(s.index, s.values))).transpose()
        logger.info(X.head())
        logger.info(X.shape)

        predictions = model.predict_proba(X)
        logger.info(predictions.shape)
        predictions_df = pd.DataFrame(
            predictions, columns=[not_treat_score_col_name, treat_score_col_name, unknown_score_col_name]
        )
        logger.info(predictions_df.head(3))

        result_df = pd.concat([partition_df, predictions_df], axis=1)
        for row in result_df.to_dict("records"):
            yield Row(**row)

    if "ARGO_NODE_ID" not in os.environ:
        transformed = transformed.repartition(
            20
        )  # Adjust based on data size - 20 is small, but okay for local development on subset of data
    else:
        logger.info(f"Number of rows remaining: {transformed.count()}")
    data = transformed.repartition(1000).rdd.mapPartitionsWithIndex(predict_partition).toDF()
    # data = data.join(
    #     transformed.select("__index_level_0__", not_treat_score_col_name, treat_score_col_name, unknown_score_col_name),
    #     on="__index_level_0__",
    #     how="inner",
    # ).cache()

    data.show()
    # 4. Validate the code at least reaches this point without OOM, as the next steps are a bit risky.
    # Example:
    # As an alternative to the below: you could use monotonically_increasing_id,
    # WITH a mapPartitionsWithIndex, so that for each partition, you extract the
    # max value (based on the index**32 IIRC), and normalize wrt that value.
    # window_spec = Window.orderBy(F.desc(treat_score_col_name))
    # data = data.orderBy(F.desc(treat_score_col_name))
    # Use row_number() for consistent sequential ranking
    # windowSpec = Window.orderBy(F.desc(treat_score_col_name))
    # # Add rank column
    # data = data.withColumn("rank", F.row_number().over(windowSpec))
    # # Add quantile rank column
    # data = data.withColumn("quantile_rank", F.percent_rank().over(windowSpec))

    return data


def generate_summary_metadata(matrix_parameters: Dict) -> pd.DataFrame:
    """
    Generate metadata for the output matrix.

    Args:
        matrix_parameters (Dict): Dictionary containing matrix parameters.

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
    summary_metadata.update({"master_filter": tags_col_names["master"]["legend"]})

    # Add metadata for statistical columns
    for stat, description in stats_col_names["per_disease"]["top"].items():
        summary_metadata[f"{stat}_top_per_disease"] = f"{description} in the top n_reporting pairs"
    for stat, description in stats_col_names["per_disease"]["all"].items():
        summary_metadata[f"{stat}_all_per_disease"] = f"{description} in all pairs"

    return pd.DataFrame(list(summary_metadata.items()), columns=["Key", "Value"])


def _process_top_pairs(
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
        "kg_name": {row["id"]: row["name"] for _, row in drugs.iterrows()},
        "list_id": {row["id"]: row["id"] for _, row in drugs.iterrows()},
        "list_name": {row["id"]: row["name"] for _, row in drugs.iterrows()},
    }

    disease_mappings = {
        "kg_name": {row["id"]: row["name"] for _, row in diseases.iterrows()},
        "list_id": {row["id"]: row["id"] for _, row in diseases.iterrows()},
        "list_name": {row["id"]: row["name"] for _, row in diseases.iterrows()},
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


def _add_descriptive_stats(
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


def _flag_known_pairs(top_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Flag known positive and negative pairs in the top pairs DataFrame.

    Args:
        top_pairs (pd.DataFrame): DataFrame containing the top pairs.

    Returns:
        pd.DataFrame: DataFrame with added flags for known positive and negative pairs.
    """
    top_pairs["is_known_positive"] = top_pairs[
        "is_known_positive"
    ]  # | top_pairs["trial_sig_better"] | top_pairs["trial_non_sig_better"]
    top_pairs["is_known_negative"] = top_pairs[
        "is_known_negative"
    ]  # | top_pairs["trial_sig_worse"] | top_pairs["trial_non_sig_worse"]
    return top_pairs


def _reorder_columns(top_pairs: pd.DataFrame, score_col_name: str, matrix_params: Dict) -> pd.DataFrame:
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
    tag_columns = (
        list(["master_filter"])
        + list(tags["drugs"].keys())
        + list(tags["diseases"].keys())
        + list(tags["pairs"].keys())
    )
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


def _apply_condition(top_pairs: pd.DataFrame, condition: List[str]) -> pd.Series:
    """Apply a single condition to the top_pairs DataFrame."""
    valid_columns = [col for col in condition if col in top_pairs.columns]
    if not valid_columns:
        return pd.Series([False] * len(top_pairs))
    return top_pairs[valid_columns].all(axis=1)


def _add_master_filter(top_pairs: pd.DataFrame, matrix_params: Dict) -> pd.DataFrame:
    """Add master_filter tag to the top_pairs DataFrame."""
    conditions = matrix_params["master"]["conditions"]
    condition_results = [_apply_condition(top_pairs, cond) for cond in conditions]
    top_pairs["master_filter"] = pd.DataFrame(condition_results).any(axis=0)
    return top_pairs


def _add_tags(
    top_pairs: pd.DataFrame, drugs: pd.DataFrame, diseases: pd.DataFrame, matrix_params: Dict
) -> pd.DataFrame:
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
    for set, set_id, df, df_id in [
        ("drugs", "kg_drug_id", drugs, "id"),
        ("diseases", "kg_disease_id", diseases, "id"),
    ]:
        for tag_name, _ in matrix_params.get(set, {}).items():
            if tag_name not in df.columns:
                logger.warning(f"Tag column '{tag_name}' not found in {set} DataFrame. Skipping.")
            else:
                tag_mapping = dict(zip(df[df_id], df[tag_name]))
                # Add the tag to top_pairs
                top_pairs[tag_name] = top_pairs[set_id].map(tag_mapping)

    top_pairs = _add_master_filter(top_pairs, matrix_params)
    return top_pairs


def generate_metadata(
    matrix_report: pd.DataFrame,
    data: pd.DataFrame,
    score_col_name: str,
    matrix_params: Dict,
    run_metadata: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a metadata report.

    Args:
        matrix_report: pd.DataFrame, matrix report dataset.
        data: pd.DataFrame, full matrix.
        score_col_name: Probability score column name.
        matrix_params: Dictionary of column names and their descriptions.
        run_metadata: Dictionary of run metadata.
    Returns:
        Tuple containing:
        - Dataframe containing metadata such as data sources version, timestamp, run name etc.
        - Dataframe with metadata about the output matrix columns.
    """
    meta_dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d"),
    }
    for key, value in run_metadata.items():
        if key == "versions":
            for subkey, subvalue in value.items():
                meta_dict[f"{subkey}_version"] = subvalue["version"]
        else:
            meta_dict[key] = value

    # Generate legends column and filter out based on
    legends_df = generate_summary_metadata(matrix_params)
    legends_df = legends_df.loc[legends_df["Key"].isin(matrix_report.columns.values)]

    # Generate metadata df
    version_df = pd.DataFrame(list(meta_dict.items()), columns=["Key", "Value"])

    # Calculate mean/median/quantile score for the full matrix
    stats_dict = {"stats_type": [], "value": []}
    for main_key in matrix_params["stats_col_names"]["full"].keys():
        # Top n stats
        stats_dict["stats_type"].append(f"{main_key}_top_n")
        stats_dict["value"].append(getattr(matrix_report[score_col_name], main_key)())
        # Full matrix stats
        stats_dict["stats_type"].append(f"{main_key}_full_matrix")
        stats_dict["value"].append(getattr(data[score_col_name], main_key)())
    # Concatenate version and legends dfs
    return version_df, pd.DataFrame(stats_dict), legends_df


def generate_report(
    data: pd.DataFrame,
    n_reporting: int,
    drugs: pd.DataFrame,
    diseases: pd.DataFrame,
    score_col_name: str,
    matrix_params: Dict,
    run_metadata: Dict,
) -> List[pd.DataFrame]:
    """Generates a report with the top pairs and metadata.

    Args:
        data: Pairs dataset.
        n_reporting: Number of pairs in the report.
        drugs: Dataframe containing names and IDs for the list of drugs.
        diseases: Dataframe containing names and IDs for the list of diseases.
        score_col_name: Probability score column name.
        matrix_params: Dictionary containing matrix metadata and other meters.
        run_metadata: Dictionary containing run metadata.
    Returns:
        Dataframe with the top pairs and additional information for the drugs and diseases.
    """
    # Add tags and process top pairs
    # Sort by the probability score
    sorted_data = data.sort_values(by=score_col_name, ascending=False)
    # Add rank and quantile rank columns
    sorted_data["rank"] = range(1, len(sorted_data) + 1)
    sorted_data["quantile_rank"] = sorted_data["rank"] / len(sorted_data)

    stats = matrix_params.get("stats_col_names")
    tags = matrix_params.get("tags")
    top_pairs = _process_top_pairs(sorted_data, n_reporting, drugs, diseases, score_col_name)
    top_pairs = _add_descriptive_stats(top_pairs, sorted_data, stats, score_col_name)
    top_pairs = _flag_known_pairs(top_pairs)
    top_pairs = _add_tags(top_pairs, drugs, diseases, tags)
    top_pairs = _reorder_columns(top_pairs, score_col_name, matrix_params)
    versions, stats, legends = generate_metadata(top_pairs, sorted_data, score_col_name, matrix_params, run_metadata)
    return {
        "metadata": versions,
        "statistics": stats,
        "legend": legends,
        "matrix": top_pairs,
    }
