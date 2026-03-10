import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List, Optional

import pyspark.sql as ps
import pyspark.sql.functions as sf
from bmt import toolkit
from matrix_pandera.schemas import get_unioned_edge_schema
from matrix_pandera.validator import Column, DataFrameSchema, check_output
from pyspark.sql import types as T

tk = toolkit.Toolkit()
logger = logging.getLogger(__name__)


class Filter(ABC):
    """Base class for all filters in the matrix pipeline.

    This abstract class defines the interface that all filters must implement.
    Each filter should implement the filter() method to transform a DataFrame
    according to its specific filtering logic.
    """

    @abstractmethod
    def apply(self, df: ps.DataFrame) -> ps.DataFrame:
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


class BiolinkDeduplicateEdges(Filter):
    """Filter that deduplicates biolink edges.

    Knowledge graphs in biolink format may contain multiple edges between nodes. Where
    edges might represent predicates at various depths in the hierarchy. This filter
    deduplicates redundant edges.

    The logic leverages the path to the predicate in the hierarchy, and removes edges
    for which "deeper" paths in the hierarchy are specified. For example: there exists
    the following edges (a)-[regulates]-(b), and (a)-[negatively-regulates]-(b). Regulates
    is on the path (regulates) whereas (regulates, negatively-regulates). In this case
    negatively-regulates is "deeper" than regulates and hence (a)-[regulates]-(b) is removed.

    Args:
        df: DataFrame with biolink edges

    Returns:
        Deduplicated DataFrame
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
    def apply(self, df: ps.DataFrame) -> ps.DataFrame:
        # Only cache the minimal columns needed for the self-join — avoids carrying heavy
        # columns like source_edge_properties through an expensive shuffle.
        slim_df = (
            df.select("subject", "object", "predicate")
            .withColumn(
                "parents",
                sf.udf(get_ancestors_for_category_delimited, T.ArrayType(T.StringType()))(sf.col("predicate")),
            )
            .cache()
        )

        # Self join to find edges that are redundant
        duplicates = (
            slim_df.alias("A")
            .join(
                slim_df.alias("B"),
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
            .select("A.subject", "A.object", "A.predicate")
            .distinct()
        )

        slim_df.unpersist()

        return df.join(duplicates, on=["subject", "object", "predicate"], how="left_anti")


class KeepRowsContaining(Filter):
    """Filter that keeps only rows containing specified values in a column.

    This filter implements the logic to keep only rows where a specific column
    contains any of the values in the provided keep_list.

    The column is expected to be an array column.
    """

    def __init__(self, column: str, keep_list: Iterable[str]):
        """Initialize the filter with column and values to keep.

        Args:
            column: Name of the column to check
            keep_list: List of values to keep
        """
        self.column = column
        self.keep_list = keep_list

    def apply(self, df: ps.DataFrame) -> ps.DataFrame:
        keep_list_array = sf.array([sf.lit(x) for x in self.keep_list])
        return df.filter(sf.exists(self.column, lambda x: sf.array_contains(keep_list_array, x)))


class RemoveRowsByColumn(Filter):
    """Filter that removes rows where a column value is in a specified list."""

    def __init__(self, column: str, remove_list: Iterable[str]):
        """Initialize the filter with column and values to remove.

        Args:
            column: Name of the column to check
            remove_list: List of values to remove
        """
        self.column = column
        self.remove_list = remove_list

    def apply(self, df: ps.DataFrame) -> ps.DataFrame:
        filter_condition = ~sf.col(self.column).isin(self.remove_list)
        return df.filter(filter_condition)


class RemoveRowsByColumnOverlap(Filter):
    """Filter that removes rows where a column array overlaps with a specified list, with an optional source exclusion where the filter won't be applied."""

    def __init__(
        self,
        column: str,
        remove_list: Iterable[str],
        upstream_data_source_column: str = "upstream_data_source",
        excluded_sources: Optional[List[str]] = None,
    ):
        """Initialize the filter with column and values to remove.

        Args:
            column: Name of the column to check.
            remove_list: List of values that will be looked for in the column
            excluded_sources: List of sources that won't have the filter applied to them
        """
        self.column = column
        self.remove_list = remove_list
        self.excluded_sources = excluded_sources or []
        self.upstream_data_source_column = upstream_data_source_column

    def apply(self, df: ps.DataFrame) -> ps.DataFrame:
        filter_condition = ~sf.arrays_overlap(sf.col(self.column), sf.lit(self.remove_list))
        if self.excluded_sources:
            filter_condition = filter_condition | sf.arrays_overlap(
                sf.col(self.upstream_data_source_column), sf.lit(self.excluded_sources)
            )

        return df.filter(filter_condition)


class TriplePattern(Filter):
    """Filter that removes edges matching specific subject-predicate-object patterns.

    This filter implements the logic to remove edges that match any of the
    specified triple patterns.
    """

    def __init__(self, triples_to_exclude: Iterable[list[str]], excluded_sources: Optional[List[str]] = None):
        """Initialize the filter with triple patterns to exclude.

        Args:
            triples_to_exclude: List of triples to exclude, where each triple is
                [subject_category, predicate, object_category]
        """
        self.triples_to_exclude = triples_to_exclude
        self.excluded_sources = excluded_sources

    def apply(self, df: ps.DataFrame) -> ps.DataFrame:
        filter_condition = sf.lit(True)
        for subject_cat, predicate, object_cat in self.triples_to_exclude:
            filter_condition = filter_condition & ~(
                (sf.col("subject_category") == subject_cat)
                & (sf.col("predicate") == predicate)
                & (sf.col("object_category") == object_cat)
            )
        if self.excluded_sources:
            filter_condition = filter_condition | sf.arrays_overlap(
                sf.col("upstream_data_source"), sf.lit(self.excluded_sources)
            )
        return df.filter(filter_condition)


# All columns in the unioned_edges
_KNOWN_EDGE_SCHEMA_COLS = frozenset(get_unioned_edge_schema().columns.keys())

# Excluded from source_edge_properties column, keep at top level
_EXCLUDED_FROM_MAP = frozenset(
    {"subject", "predicate", "object", "primary_knowledge_source", "primary_knowledge_sources", "upstream_data_source"}
)


class DeduplicateEdges(Filter):
    """Collapses duplicate (subject, predicate, object) triples into a single edge.

    Qualifier and evidence fields (knowledge_level, agent_type, qualifiers, publications,
    num_references, num_sentences) are captured in source_edge_properties — a JSON string
    encoding a map keyed by primary_knowledge_source — preserving per-source detail instead
    of using F.first().

    publications, num_references, and num_sentences also remain as flat top-level aggregations
    for convenience.

    Columns added by the filtering pipeline (e.g. subject_category, object_category) are
    preserved by including them in the groupBy key; they are excluded from source_edge_properties.
    """

    def apply(self, df: ps.DataFrame) -> ps.DataFrame:
        filtering_pipeline_cols = [c for c in df.columns if c not in _KNOWN_EDGE_SCHEMA_COLS]

        source_property_cols = [
            c for c in df.columns if c not in _EXCLUDED_FROM_MAP and c not in filtering_pipeline_cols
        ]

        groupby_keys = ["subject", "predicate", "object"] + filtering_pipeline_cols

        return df.groupBy(groupby_keys).agg(
            sf.flatten(sf.collect_set("upstream_data_source")).alias("upstream_data_source"),
            sf.first("primary_knowledge_source", ignorenulls=True).alias("primary_knowledge_source"),
            sf.collect_set("primary_knowledge_source").alias("primary_knowledge_sources"),
            sf.flatten(sf.collect_set("aggregator_knowledge_source")).alias("aggregator_knowledge_source"),
            sf.flatten(sf.collect_set("publications")).alias("publications"),
            sf.max("num_references").cast(T.IntegerType()).alias("num_references"),
            sf.max("num_sentences").cast(T.IntegerType()).alias("num_sentences"),
            sf.to_json(
                sf.map_from_entries(
                    sf.collect_list(
                        sf.struct(
                            sf.col("primary_knowledge_source").alias("key"),
                            sf.struct(*[sf.col(c) for c in source_property_cols]).alias("value"),
                        )
                    )
                )
            ).alias("source_edge_properties"),
        )
