import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable

import pyspark.sql as ps
import pyspark.sql.functions as sf
from bmt import toolkit
from pyspark.sql import types as T

from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output

tk = toolkit.Toolkit()
logger = logging.getLogger(__name__)


class Filter(ABC):
    """Base class for all filters in the matrix pipeline.

    This abstract class defines the interface that all filters must implement.
    Each filter should implement the filter() method to transform a DataFrame
    according to its specific filtering logic.
    """

    @abstractmethod
    def filter(self, df: ps.DataFrame) -> ps.DataFrame:
        """Apply the filter to the input DataFrame.

        Args:
            df: Input DataFrame to filter

        Returns:
            Filtered DataFrame
        """
        pass


def get_ancestors_for_category_delimited(category: str, mixin: bool = False) -> list[str]:
    """Wrapper function to get ancestors for a category. The arguments were used to match the args used by Chunyu
    https://biolink.github.io/biolink-model-toolkit/index.html#bmt.toolkit.Toolkit.get_ancestors
    Args:
        category: Category to get ancestors for
        mixin: Whether to include mixins
    Returns:
        List of ancestors in a string format
    """
    return tk.get_ancestors(category, mixin=mixin, formatted=True, reflexive=True)


class BiolinkDeduplicateEdgesFilter(Filter):
    """Filter that deduplicates biolink edges.

    Knowledge graphs in biolink format may contain multiple edges between nodes. Where
    edges might represent predicates at various depths in the hierarchy. This filter
    deduplicates redundant edges.

    The logic leverages the path to the predicate in the hierarchy, and removes edges
    for which "deeper" paths in the hierarchy are specified. For example: there exists
    the following edges (a)-[regulates]-(b), and (a)-[negatively-regulates]-(b). Regulates
    is on the path (regulates) whereas (regulates, negatively-regulates). In this case
    negatively-regulates is "deeper" than regulates and hence (a)-[regulates]-(b) is removed.
    """

    @check_output(
        DataFrameSchema(
            columns={
                "subject": Column(T.StringType(), nullable=False),
                "predicate": Column(T.StringType(), nullable=False),
                "object": Column(T.StringType(), nullable=False),
            },
            unique=["subject", "object", "predicate"],
        ),
    )
    def filter(self, df: ps.DataFrame) -> ps.DataFrame:
        """Deduplicate biolink edges.

        Args:
            df: DataFrame with biolink edges

        Returns:
            Deduplicated DataFrame
        """
        # Enrich edges with path to predicates in biolink hierarchy
        edges_df = df.withColumn(
            "parents", sf.udf(get_ancestors_for_category_delimited, T.ArrayType(T.StringType()))(sf.col("predicate"))
        ).cache()

        # Self join to find edges that are redundant
        duplicates = (
            edges_df.alias("A")
            .join(
                edges_df.alias("B"),
                on=[
                    (sf.col("A.subject") == sf.col("B.subject"))
                    & (sf.col("A.object") == sf.col("B.object"))
                    & (sf.col("A.predicate") != sf.col("B.predicate"))
                ],
                how="left",
            )
            .withColumn(
                "subpath",
                sf.col("B.parents").isNotNull() & sf.expr("forall(A.parents, x -> array_contains(B.parents, x))"),
            )
            .filter(sf.col("subpath"))
            .select("A.*")
            .select("subject", "object", "predicate")
            .distinct()
        )

        return (
            edges_df.alias("edges")
            .join(duplicates, on=["subject", "object", "predicate"], how="left_anti")
            .select("edges.*")
        )


class KeepRowsContainingFilter(Filter):
    """Filter that keeps only rows containing specified values in a column.

    This filter implements the logic to keep only rows where a specific column
    contains any of the values in the provided keep_list. If the column is a string,
    it will be split into an array before filtering.
    """

    def __init__(self, column: str, keep_list: Iterable[str]):
        """Initialize the filter with column and values to keep.

        Args:
            column: Name of the column to check
            keep_list: List of values to keep
        """
        self.column = column
        self.keep_list = keep_list

    def filter(self, df: ps.DataFrame) -> ps.DataFrame:
        """Keep only rows where the specified column contains any value from keep_list.

        Args:
            df: Input DataFrame to filter

        Returns:
            DataFrame with only matching rows kept
        """
        keep_list_array = sf.array([sf.lit(x) for x in self.keep_list])
        return df.filter(sf.exists(self.column, lambda x: sf.array_contains(keep_list_array, x)))


class RemoveRowsByColumnFilter(Filter):
    """Filter that removes rows where a column value is in a specified list.

    This filter implements the logic to remove rows where a specific column
    contains any of the values in the provided remove_list.
    """

    def __init__(self, column: str, remove_list: Iterable[str]):
        """Initialize the filter with column and values to remove.

        Args:
            column: Name of the column to check
            remove_list: List of values to remove
        """
        self.column = column
        self.remove_list = remove_list

    def filter(self, df: ps.DataFrame) -> ps.DataFrame:
        """Remove rows where the specified column contains any value from remove_list.

        Args:
            df: Input DataFrame to filter

        Returns:
            DataFrame with matching rows removed
        """
        filter_condition = ~sf.col(self.column).isin(self.remove_list)
        return df.filter(filter_condition)


class TriplePatternFilter(Filter):
    """Filter that removes edges matching specific subject-predicate-object patterns.

    This filter implements the logic to remove edges that match any of the
    specified triple patterns.
    """

    def __init__(self, triples_to_exclude: Iterable[list[str]]):
        """Initialize the filter with triple patterns to exclude.

        Args:
            triples_to_exclude: List of triples to exclude, where each triple is
                [subject_category, predicate, object_category]
        """
        self.triples_to_exclude = triples_to_exclude

    def filter(self, df: ps.DataFrame) -> ps.DataFrame:
        """Remove edges that match any of the specified triple patterns.

        Args:
            df: Input DataFrame to filter

        Returns:
            DataFrame with matching edges removed
        """
        filter_condition = sf.lit(True)
        for subject_cat, predicate, object_cat in self.triples_to_exclude:
            filter_condition = filter_condition & ~(
                (sf.col("subject_category") == subject_cat)
                & (sf.col("predicate") == predicate)
                & (sf.col("object_category") == object_cat)
            )
        return df.filter(filter_condition)
