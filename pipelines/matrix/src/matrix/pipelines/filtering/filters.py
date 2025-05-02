import logging
from collections.abc import Iterable

import pyspark.sql as ps
import pyspark.sql.functions as sf
from bmt import toolkit
from pyspark.sql import types as T

from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output

tk = toolkit.Toolkit()
logger = logging.getLogger(__name__)


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
def biolink_deduplicate_edges(r_edges_df: ps.DataFrame) -> ps.DataFrame:
    """Function to deduplicate biolink edges.
    Knowledge graphs in biolink format may contain multiple edges between nodes. Where
    edges might represent predicates at various depths in the hierarchy. This function
    deduplicates redundant edges.
    The logic leverages the path to the predicate in the hierarchy, and removes edges
    for which "deeper" paths in the hierarchy are specified. For example: there exists
    the following edges (a)-[regulates]-(b), and (a)-[negatively-regulates]-(b). Regulates
    is on the path (regulates) whereas (regulates, negatively-regulates). In this case
    negatively-regulates is "deeper" than regulates and hence (a)-[regulates]-(b) is removed.
    Args:
        edges_df: dataframe with biolink edges
    Returns:
        Deduplicated dataframe
    """
    # Enrich edges with path to predicates in biolink hierarchy
    edges_df = r_edges_df.withColumn(
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
            "subpath", sf.col("B.parents").isNotNull() & sf.expr("forall(A.parents, x -> array_contains(B.parents, x))")
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


def keep_rows_containing(
    input_df: ps.DataFrame,
    keep_list: Iterable[str],
    column: str,
    **kwargs,
) -> ps.DataFrame:
    """Function to only keep rows containing a category."""
    keep_list_array = sf.array([sf.lit(x) for x in keep_list])
    return input_df.filter(sf.exists(column, lambda x: sf.array_contains(keep_list_array, x)))
