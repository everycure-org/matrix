"""Module containing strategies for reporting tables node"""

import abc
from datetime import datetime

import pyspark.sql as ps
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType


class ReportingTableGenerator(abc.ABC):
    """Class representing generators outputting tables for the matrix generation pipeline."""

    def __init__(self, name: str) -> None:
        """Initializes a ReportingTableGenerator instance.

        Args:
            name: Name assigned to the table
        """
        self.name = name

    @abc.abstractmethod
    def generate(
        self, sorted_matrix_df: ps.DataFrame, drugs_df: ps.DataFrame, diseases_df: ps.DataFrame
    ) -> ps.DataFrame:
        """Generate a table.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list

        Returns:
            Spark DataFrame containing the table
        """
        pass


class MatrixRunInfo(ReportingTableGenerator):
    """Class for generating a table containing information about the matrix run."""

    def __init__(
        self,
        name: str,
        versions: dict,
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.versions = versions
        self.kwargs = kwargs

    def generate(
        self,
        sorted_matrix_df: ps.DataFrame,
        drugs_df: ps.DataFrame,
        diseases_df: ps.DataFrame,
    ) -> ps.DataFrame:
        """Generate a table.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix (not used)
            drugs_df: DataFrame containing the drugs list (not used)
            diseases_df: DataFrame containing the diseases list (not used)
            versions: Versions of the data sources
            kwargs: All other information. Injected through parameter config.

        Returns:
            Spark DataFrame containing the table
        """
        # Create Spark session
        spark = SparkSession.builder.getOrCreate()

        # Create list of key-value pairs
        metadata_list = [
            ("timestamp", datetime.now().strftime("%Y-%m-%d")),
        ]
        metadata_list.extend([(f"{key}_version", value["version"]) for key, value in self.versions.items()])
        metadata_list.extend([(k, str(v)) for k, v in self.kwargs.items()])

        # Return Spark DataFrame
        schema = StructType(
            [
                StructField("key", StringType(), True),
                StructField("value", StringType(), True),
            ]
        )
        return spark.createDataFrame(metadata_list, schema=schema)


class TopPairs(ReportingTableGenerator):
    """Class for generating a table containing the top pairs."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def generate(
        self,
        sorted_matrix_df: ps.DataFrame,
        drugs_df: ps.DataFrame,
        diseases_df: ps.DataFrame,
        n_reporting: int,
        score_col: str,
    ) -> ps.DataFrame:
        """Generate a table.

        TODO don't forget raise if n_reporting too large

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list
            n_reporting: Number of pairs to report
            score_col: Score column

        Returns:
            Spark DataFrame containing the table
        """
        return  # TODO: Implement


class RankToScore(ReportingTableGenerator):
    """Class for generating a table containing the rank to score."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def generate(
        self,
        sorted_matrix_df: ps.DataFrame,
        drugs_df: ps.DataFrame,
        diseases_df: ps.DataFrame,
        ranks_lst: list[int],
        score_col: str,
    ) -> ps.DataFrame:
        """Generate a table.

        TODO don't forget raise if max ranks_lst too large

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list
            ranks_lst: List of ranks
            score_col: Score column

        Returns:
            Spark DataFrame containing the table
        """
        return  # TODO: Implement


class TopFrequentFlyers(ReportingTableGenerator):
    """Class for generating a table containing the top frequent flyers."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def generate(
        self,
        sorted_matrix_df: ps.DataFrame,
        drugs_df: ps.DataFrame,
        diseases_df: ps.DataFrame,
        specific_col: str,
        specific_entity_name: str,
        count_in_n_lst: list[int],
        sort_by_col: str,
        score_col: str,
    ) -> ps.DataFrame:
        """Generate a table.

        TODO don't forget raise if sort_by_col not valid

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list
            specific_col: Column to count frequency in
            specific_entity_name: Name of the specific entity
            count_in_n_lst: List of counts to count frequency in
            sort_by_col: Column to sort by
            score_col: Score column

        Returns:
            Spark DataFrame containing the table
        """
        return  # TODO: Implement


# TODO: Remove - reference code

# def generate_report(
#     data: pd.DataFrame,
#     n_reporting: int,
#     drugs: pd.DataFrame,
#     diseases: pd.DataFrame,
#     score_col_name: str,
#     matrix_params: Dict,
#     run_metadata: Dict,
# ) -> List[pd.DataFrame]:
#     """Generates a report with the top pairs and metadata.

#     Args:
#         data: Pairs dataset.
#         n_reporting: Number of pairs in the report.
#         drugs: Dataframe containing names and IDs for the list of drugs.
#         diseases: Dataframe containing names and IDs for the list of diseases.
#         score_col_name: Probability score column name.
#         matrix_params: Dictionary containing matrix metadata and other meters.
#         run_metadata: Dictionary containing run metadata.
#     Returns:
#         Dataframe with the top pairs and additional information for the drugs and diseases.
#     """
#     # Add tags and process top pairs
#     stats = matrix_params.get("stats_col_names")
#     tags = matrix_params.get("tags")
#     top_pairs = _process_top_pairs(data, n_reporting, drugs, diseases, score_col_name)
#     top_pairs = _add_descriptive_stats(top_pairs, data, stats, score_col_name)
#     top_pairs = _flag_known_pairs(top_pairs)
#     top_pairs = _add_tags(top_pairs, drugs, diseases, tags)
#     top_pairs = _reorder_columns(top_pairs, score_col_name, matrix_params)
#     versions, stats, legends = generate_metadata(top_pairs, data, score_col_name, matrix_params, run_metadata)
#     return {
#         "metadata": versions,
#         "statistics": stats,
#         "legend": legends,
#         "matrix": top_pairs,
#     }

# def generate_summary_metadata(matrix_parameters: Dict) -> pd.DataFrame:
#     """
#     Generate metadata for the output matrix.

#     Args:
#         matrix_parameters (Dict): Dictionary containing matrix parameters.

#     Returns:
#         pd.DataFrame: DataFrame containing summary metadata.
#     """
#     summary_metadata = {}

#     meta_col_names = matrix_parameters["metadata"]
#     stats_col_names = matrix_parameters["stats_col_names"]
#     tags_col_names = matrix_parameters["tags"]

#     # Add metadata for ID columns
#     summary_metadata.update(meta_col_names["drug_list"])
#     summary_metadata.update(meta_col_names["disease_list"])

#     # Add metadata for KG columns and tags
#     summary_metadata.update(meta_col_names["kg_data"])

#     # Add metadata for tags and filters
#     summary_metadata.update(tags_col_names["drugs"])
#     summary_metadata.update(tags_col_names["pairs"])
#     summary_metadata.update(tags_col_names["diseases"])
#     summary_metadata.update({"master_filter": tags_col_names["master"]["legend"]})

#     # Add metadata for statistical columns
#     for stat, description in stats_col_names["per_disease"]["top"].items():
#         summary_metadata[f"{stat}_top_per_disease"] = f"{description} in the top n_reporting pairs"
#     for stat, description in stats_col_names["per_disease"]["all"].items():
#         summary_metadata[f"{stat}_all_per_disease"] = f"{description} in all pairs"

#     return pd.DataFrame(list(summary_metadata.items()), columns=["Key", "Value"])


# def _process_top_pairs(
#     data: pd.DataFrame, n_reporting: int, drugs: pd.DataFrame, diseases: pd.DataFrame, score_col_name: str
# ) -> pd.DataFrame:
#     """
#     Process the top pairs from the data and add additional information.

#     Args:
#         data (pd.DataFrame): The input DataFrame containing all pairs.
#         n_reporting (int): The number of top pairs to process.
#         drugs (pd.DataFrame): DataFrame containing drug information.
#         diseases (pd.DataFrame): DataFrame containing disease information.
#         score_col_name (str): The name of the column containing the score.

#     Returns:
#         pd.DataFrame: Processed DataFrame containing the top pairs with additional information.
#     """
#     top_pairs = data.head(n_reporting).copy()
#     # Generate mapping dictionaries
#     drug_mappings = {
#         "kg_name": {row["id"]: row["name"] for _, row in drugs.iterrows()},
#         "list_id": {row["id"]: row["id"] for _, row in drugs.iterrows()},
#         "list_name": {row["id"]: row["name"] for _, row in drugs.iterrows()},
#     }

#     disease_mappings = {
#         "kg_name": {row["id"]: row["name"] for _, row in diseases.iterrows()},
#         "list_id": {row["id"]: row["id"] for _, row in diseases.iterrows()},
#         "list_name": {row["id"]: row["name"] for _, row in diseases.iterrows()},
#     }

#     # Add additional information
#     top_pairs["kg_drug_name"] = top_pairs["source"].map(drug_mappings["kg_name"])
#     top_pairs["kg_disease_name"] = top_pairs["target"].map(disease_mappings["kg_name"])
#     top_pairs["disease_id"] = top_pairs["target"].map(disease_mappings["list_id"])
#     top_pairs["disease_name"] = top_pairs["target"].map(disease_mappings["list_name"])
#     top_pairs["drug_id"] = top_pairs["source"].map(drug_mappings["list_id"])
#     top_pairs["drug_name"] = top_pairs["source"].map(drug_mappings["list_name"])

#     # Rename ID columns and add pair ID
#     top_pairs = top_pairs.rename(columns={"source": "kg_drug_id", "target": "kg_disease_id"})
#     top_pairs["pair_id"] = top_pairs["drug_id"] + "|" + top_pairs["disease_id"]

#     return top_pairs


# def _add_descriptive_stats(
#     top_pairs: pd.DataFrame, data: pd.DataFrame, stats_col_names: Dict, score_col_name: str
# ) -> pd.DataFrame:
#     """
#     Add descriptive statistics to the top pairs DataFrame.

#     Args:
#         top_pairs (pd.DataFrame): DataFrame containing the top pairs.
#         data (pd.DataFrame): The full dataset containing all pairs.
#         stats_col_names (Dict): Dictionary containing the names of statistical columns.
#         score_col_name (str): The name of the column containing the score.

#     Returns:
#         pd.DataFrame: DataFrame with added descriptive statistics.
#     """
#     data = data.sort_values(by=score_col_name, ascending=False)
#     for entity_type, df_col, id_col in [("disease", "target", "kg_disease_id"), ("drug", "source", "kg_drug_id")]:
#         # Calculate stats for top pairs
#         for stat in stats_col_names[f"per_{entity_type}"]["top"].keys():
#             top_pairs[f"{stat}_top_per_{entity_type}"] = top_pairs.groupby(id_col)[score_col_name].transform(stat)

#         # Calculate stats for all pairs (need to use different df)
#         all_pairs = data[data[df_col].isin(top_pairs[id_col].unique())]
#         for stat in stats_col_names[f"per_{entity_type}"]["all"].keys():
#             stat_dict = all_pairs.groupby(df_col)[score_col_name].agg(stat)
#             top_pairs[f"{stat}_all_per_{entity_type}"] = top_pairs[id_col].map(stat_dict)
#     return top_pairs


# def _flag_known_pairs(top_pairs: pd.DataFrame) -> pd.DataFrame:
#     """
#     Flag known positive and negative pairs in the top pairs DataFrame.

#     Args:
#         top_pairs (pd.DataFrame): DataFrame containing the top pairs.

#     Returns:
#         pd.DataFrame: DataFrame with added flags for known positive and negative pairs.
#     """
#     top_pairs["is_known_positive"] = top_pairs[
#         "is_known_positive"
#     ]  # | top_pairs["trial_sig_better"] | top_pairs["trial_non_sig_better"]
#     top_pairs["is_known_negative"] = top_pairs[
#         "is_known_negative"
#     ]  # | top_pairs["trial_sig_worse"] | top_pairs["trial_non_sig_worse"]
#     return top_pairs


# def _reorder_columns(top_pairs: pd.DataFrame, score_col_name: str, matrix_params: Dict) -> pd.DataFrame:
#     """
#     Reorder columns in the top pairs DataFrame.

#     Args:
#         top_pairs (pd.DataFrame): DataFrame containing the top pairs.
#         score_col_name (str): The name of the column containing the score.
#         matrix_params (Dict): Dictionary containing matrix parameters.

#     Returns:
#         pd.DataFrame: DataFrame with reordered columns.
#     """
#     meta_col_names = matrix_params["metadata"]
#     stats_col_names = matrix_params["stats_col_names"]
#     tags = matrix_params["tags"]

#     id_columns = list(meta_col_names["drug_list"].keys()) + list(meta_col_names["disease_list"].keys())
#     score_columns = [score_col_name]
#     tag_columns = (
#         list(["master_filter"])
#         + list(tags["drugs"].keys())
#         + list(tags["diseases"].keys())
#         + list(tags["pairs"].keys())
#     )
#     kg_columns = list(meta_col_names["kg_data"].keys())
#     stat_columns = list()
#     for main_key in ["per_disease", "per_drug"]:
#         for sub_key in ["top", "all"]:
#             stat_columns = stat_columns + [
#                 f"{stat}_{sub_key}_{main_key}" for stat in list(stats_col_names[main_key][sub_key].keys())
#             ]
#     columns_order = id_columns + score_columns + kg_columns + tag_columns + stat_columns

#     # Remove columns that are not in the top pairs DataFrame but are specified in params
#     columns_order = [col for col in columns_order if col in top_pairs.columns]
#     return top_pairs[columns_order]


# def _apply_condition(top_pairs: pd.DataFrame, condition: List[str]) -> pd.Series:
#     """Apply a single condition to the top_pairs DataFrame."""
#     valid_columns = [col for col in condition if col in top_pairs.columns]
#     if not valid_columns:
#         return pd.Series([False] * len(top_pairs))
#     return top_pairs[valid_columns].all(axis=1)


# def _add_master_filter(top_pairs: pd.DataFrame, matrix_params: Dict) -> pd.DataFrame:
#     """Add master_filter tag to the top_pairs DataFrame."""
#     conditions = matrix_params["master"]["conditions"]
#     condition_results = [_apply_condition(top_pairs, cond) for cond in conditions]
#     top_pairs["master_filter"] = pd.DataFrame(condition_results).any(axis=0)
#     return top_pairs


# def _add_tags(
#     top_pairs: pd.DataFrame, drugs: pd.DataFrame, diseases: pd.DataFrame, matrix_params: Dict
# ) -> pd.DataFrame:
#     """
#     Add tag columns to the top pairs DataFrame.

#     Args:
#         top_pairs (pd.DataFrame): DataFrame containing the top pairs.
#         drugs (pd.DataFrame): DataFrame containing drug information.
#         diseases (pd.DataFrame): DataFrame containing disease information.
#         matrix_params (Dict): Dictionary containing matrix parameters.

#     Returns:
#         pd.DataFrame: DataFrame with added tag columns.
#     """
#     # Add tag columns for drugs and diseasesto the top pairs DataFrame
#     for set, set_id, df, df_id in [
#         ("drugs", "kg_drug_id", drugs, "id"),
#         ("diseases", "kg_disease_id", diseases, "id"),
#     ]:
#         for tag_name, _ in matrix_params.get(set, {}).items():
#             if tag_name not in df.columns:
#                 logger.warning(f"Tag column '{tag_name}' not found in {set} DataFrame. Skipping.")
#             else:
#                 tag_mapping = dict(zip(df[df_id], df[tag_name]))
#                 # Add the tag to top_pairs
#                 top_pairs[tag_name] = top_pairs[set_id].map(tag_mapping)

#     top_pairs = _add_master_filter(top_pairs, matrix_params)
#     return top_pairs


# def generate_metadata(
#     matrix_report: pd.DataFrame,
#     data: pd.DataFrame,
#     score_col_name: str,
#     matrix_params: Dict,
#     run_metadata: Dict,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Generates a metadata report.

#     Args:
#         matrix_report: pd.DataFrame, matrix report dataset.
#         data: pd.DataFrame, full matrix.
#         score_col_name: Probability score column name.
#         matrix_params: Dictionary of column names and their descriptions.
#         run_metadata: Dictionary of run metadata.
#     Returns:
#         Tuple containing:
#         - Dataframe containing metadata such as data sources version, timestamp, run name etc.
#         - Dataframe with metadata about the output matrix columns.
#     """
#     meta_dict = {
#         "timestamp": datetime.now().strftime("%Y-%m-%d"),
#     }
#     for key, value in run_metadata.items():
#         if key == "versions":
#             for subkey, subvalue in value.items():
#                 meta_dict[f"{subkey}_version"] = subvalue["version"]
#         else:
#             meta_dict[key] = value

#     # Generate legends column and filter out based on
#     legends_df = generate_summary_metadata(matrix_params)
#     legends_df = legends_df.loc[legends_df["Key"].isin(matrix_report.columns.values)]

#     # Generate metadata df
#     version_df = pd.DataFrame(list(meta_dict.items()), columns=["Key", "Value"])

#     # Calculate mean/median/quantile score for the full matrix
#     stats_dict = {"stats_type": [], "value": []}
#     for main_key in matrix_params["stats_col_names"]["full"].keys():
#         # Top n stats
#         stats_dict["stats_type"].append(f"{main_key}_top_n")
#         stats_dict["value"].append(getattr(matrix_report[score_col_name], main_key)())
#         # Full matrix stats
#         stats_dict["stats_type"].append(f"{main_key}_full_matrix")
#         stats_dict["value"].append(getattr(data[score_col_name], main_key)())
#     # Concatenate version and legends dfs
#     return version_df, pd.DataFrame(stats_dict), legends_df
