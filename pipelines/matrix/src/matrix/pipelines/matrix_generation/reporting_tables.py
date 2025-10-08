"""Module containing strategies for reporting tables node"""

import abc
from datetime import datetime

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F


class ReportingTableGenerator(abc.ABC):
    """Class representing generators outputting tables for the matrix generation pipeline.

    The generate method receives spark dataframes for fast computation and returns a pandas dataframe for easy-to-view output.
    """

    def __init__(self, name: str) -> None:
        """Initializes a ReportingTableGenerator instance.

        Args:
            name: Name assigned to the table (used as filename)
        """
        self.name = name

    @abc.abstractmethod
    def generate(
        self, sorted_matrix_df: ps.DataFrame, drugs_df: ps.DataFrame, diseases_df: ps.DataFrame
    ) -> pd.DataFrame:
        """Generate a table.

        This output is a pandas dataframe for easy-to-view output in MLFlow.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list

        Returns:
            Pandas DataFrame containing the table
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
        """Initializes a MatrixRunInfo instance.

        Args:
            name: Name assigned to the table
            versions: Versions of the data sources
            kwargs: All other information. Injected through parameters config.
        """
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

        Returns:
            Spark DataFrame containing the table
        """
        metadata_list = [
            ("timestamp", datetime.now().strftime("%Y-%m-%d")),
        ]
        metadata_list.extend([(f"{key}_version", value["version"]) for key, value in self.versions.items()])
        metadata_list.extend([(k, str(v)) for k, v in self.kwargs.items()])
        return pd.DataFrame(metadata_list, columns=["key", "value"])


class TopPairs(ReportingTableGenerator):
    """Class for generating a table containing the top pairs according to a score column."""

    def __init__(
        self,
        name: str,
        n_reporting: int,
        score_col: str,
        columns_to_keep: list[str] = ["treat score", "not treat score", "unknown score"],
    ) -> None:
        """Initializes a TopPairs instance.

        Args:
            name: Name assigned to the table
            n_reporting: Number of pairs to report
            score_col: Score column to sort by
        """
        super().__init__(name)
        self.n_reporting = n_reporting
        self.score_col = score_col
        self.columns_to_keep = columns_to_keep

    def generate(
        self,
        matrix_df: ps.DataFrame,
        drugs_df: ps.DataFrame,
        diseases_df: ps.DataFrame,
    ) -> pd.DataFrame:
        """Generate a table.

        Args:
            matrix_df: DataFrame containing the drug-disease pair scores
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list

        Returns:
            Pandas DataFrame containing the table
        """
        # Extract top pairs and join names
        top_pairs = (
            matrix_df.orderBy(self.score_col, ascending=False)
            .limit(self.n_reporting)
            .select(
                F.col("translator_id_source").alias("drug_id"),
                F.col("target").alias("disease_id"),
                *self.columns_to_keep,
            )
            .join(
                drugs_df.select(F.col("id").alias("drug_id"), F.col("name").alias("drug_name")),
                how="left",
                on="drug_id",
            )
            .join(
                diseases_df.select(F.col("id").alias("disease_id"), F.col("name").alias("disease_name")),
                how="left",
                on="disease_id",
            )
        )

        # Reorder columns and return
        return top_pairs.select("drug_name", "disease_name", "drug_id", "disease_id", *self.columns_to_keep).toPandas()


class RankToScore(ReportingTableGenerator):
    """Class for generating a table containing the rank to score."""

    def __init__(
        self,
        name: str,
        ranks_lst: list[int],
        score_col: str,
    ) -> None:
        """Initializes a RankToScore instance.

        Args:
            name: Name assigned to the table
            ranks_lst: List of ranks to show in the table
            score_col: Score column

        """
        super().__init__(name)
        self.ranks_lst = ranks_lst
        self.score_col = score_col

    def generate(
        self,
        sorted_matrix_df: ps.DataFrame,
        drugs_df: ps.DataFrame,
        diseases_df: ps.DataFrame,
    ) -> pd.DataFrame:
        """Generate a table.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list (not used)
            diseases_df: DataFrame containing the diseases list (not used)

        Returns:
            Spark DataFrame containing the table
        """
        if max(self.ranks_lst) > sorted_matrix_df.count():
            raise ValueError(f"max ranks_lst is too large: {max(self.ranks_lst)} > {sorted_matrix_df.count()}")

        return sorted_matrix_df.filter(F.col("rank").isin(self.ranks_lst)).select("rank", self.score_col).toPandas()


class TopFrequentFlyers(ReportingTableGenerator):
    """Class for generating a table containing the top frequent flyers."""

    def __init__(
        self,
        name: str,
        is_drug_mode: bool,
        count_in_n_lst: list[int],
        sort_by_col: str,
        score_col: str,
    ) -> None:
        """Initializes a TopFrequentFlyers instance.

        Args:
            name: Name assigned to the table
            is_drug_mode: True for drug frequent flyers, False for disease frequent flyers
            count_in_n_lst: List of counts to count frequency in
            sort_by_col: Column to sort by
            score_col: Score column
        """
        super().__init__(name)
        self.is_drug_mode = is_drug_mode
        self.count_in_n_lst = count_in_n_lst
        self.sort_by_col = sort_by_col
        self.score_col = score_col

    def generate(
        self,
        sorted_matrix_df: ps.DataFrame,
        drugs_df: ps.DataFrame,
        diseases_df: ps.DataFrame,
    ) -> pd.DataFrame:
        """Generate a table.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list

        Returns:
            Spark DataFrame containing the table
        """
        # Raise error if sort_by_col not valid
        valid_sort_by_cols = [f"count_in_{n}" for n in self.count_in_n_lst] + [
            "mean",
            "max",
        ]
        if self.sort_by_col not in valid_sort_by_cols:
            raise ValueError(f"sort_by_col must be one of the following: {valid_sort_by_cols}")

        # Set up drug or disease mode
        if self.is_drug_mode:
            entity_df = drugs_df.select(F.col("id").alias("id"), F.col("name").alias("name"))
            specific_col = "translator_id_source"
        else:
            entity_df = diseases_df.select(F.col("id").alias("id"), F.col("name").alias("name"))
            specific_col = "target"

        # Create table with mean and max scores for each entity
        frequent_flyer_table = entity_df.join(
            sorted_matrix_df.groupBy(specific_col)
            .agg(F.mean(self.score_col).alias("mean"))
            .withColumnRenamed(specific_col, "id"),
            on="id",
            how="left",
        ).join(
            sorted_matrix_df.groupBy(specific_col)
            .agg(F.max(self.score_col).alias("max"))
            .withColumnRenamed(specific_col, "id"),
            on="id",
            how="left",
        )

        # Count frequency in top n
        for n in self.count_in_n_lst:
            frequent_flyer_table = frequent_flyer_table.join(
                sorted_matrix_df.limit(n)
                .groupBy(specific_col)
                .agg(F.count(specific_col).alias(f"count_in_{n}"))
                .withColumnRenamed(specific_col, "id"),
                on="id",
                how="left",
            ).fillna(0)

        # Return table sorted by chosen column
        return frequent_flyer_table.orderBy(F.col(self.sort_by_col).desc()).toPandas()
