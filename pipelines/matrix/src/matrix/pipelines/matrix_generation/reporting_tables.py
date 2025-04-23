"""Module containing strategies for reporting tables node"""

import abc
from datetime import datetime

import pyspark.sql as ps
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
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
            score_col: Score column
        """
        super().__init__(name)
        self.n_reporting = n_reporting
        self.score_col = score_col
        self.columns_to_keep = columns_to_keep

    def generate(
        self,
        sorted_matrix_df: ps.DataFrame,
        drugs_df: ps.DataFrame,
        diseases_df: ps.DataFrame,
    ) -> ps.DataFrame:
        """Generate a table.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list

        Returns:
            Spark DataFrame containing the table
        """
        if self.n_reporting > sorted_matrix_df.count():
            raise ValueError(f"n_reporting is too large: {self.n_reporting} > {sorted_matrix_df.count()}")

        # Extract top pairs and join names
        top_pairs = (
            sorted_matrix_df.orderBy(self.score_col, ascending=False)
            .limit(self.n_reporting)
            .select(
                col("source").alias("drug_id"),
                col("target").alias("disease_id"),
                *self.columns_to_keep,
            )
            .join(
                drugs_df.select(col("curie").alias("drug_id"), col("name").alias("drug_name")),
                how="left",
                on="drug_id",
            )
            .join(
                diseases_df.select(col("category_class").alias("disease_id"), col("label").alias("disease_name")),
                how="left",
                on="disease_id",
            )
        )

        # Reorder columns and return
        return top_pairs.select(
            "drug_name", "disease_name", *[col for col in top_pairs.columns if col not in ["drug_name", "disease_name"]]
        )


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
