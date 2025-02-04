import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.functions as f
from bmt import toolkit
from pyspark.sql import types as T

from matrix.utils.pa_utils import Column, DataFrameSchema, check_output

tk = toolkit.Toolkit()

logger = logging.getLogger(__name__)


def get_ancestors_for_category_delimited(category: str, mixin: bool = False) -> List[str]:
    """Wrapper function to get ancestors for a category. The arguments were used to match the args used by Chunyu
    https://biolink.github.io/biolink-model-toolkit/index.html#bmt.toolkit.Toolkit.get_ancestors

    Args:
        category: Category to get ancestors for
        formatted: Whether to format element names as curies
        mixin: Whether to include mixins
        reflexive: Whether to include query element in the list
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
        "parents", F.udf(get_ancestors_for_category_delimited, T.ArrayType(T.StringType()))(F.col("predicate"))
    )
    # Self join to find edges that are redundant
    duplicates = (
        edges_df.alias("A")
        .join(
            edges_df.alias("B"),
            on=[
                (f.col("A.subject") == f.col("B.subject"))
                & ((f.col("A.object") == f.col("B.object")) & (f.col("A.predicate") != f.col("B.predicate")))
            ],
            how="left",
        )
        .withColumn(
            "subpath", f.col("B.parents").isNotNull() & f.expr("forall(A.parents, x -> array_contains(B.parents, x))")
        )
        .filter(f.col("subpath"))
        .select("A.*")
        .select("subject", "object", "predicate")
        .distinct()
        .withColumn("is_redundant", f.lit(True))
    )
    return (
        edges_df.alias("edges")
        .join(duplicates, on=["subject", "object", "predicate"], how="left")
        .filter(F.col("is_redundant").isNull())
        .select("edges.*")
    )


@check_output(
    DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "category": Column(T.StringType(), nullable=False),
        },
        unique=["id"],
    ),
)
def determine_most_specific_category(nodes: ps.DataFrame) -> ps.DataFrame:
    """Function to retrieve most specific entry for each node.

    This function uses the `all_categories` column to infer a final `category` for the
    node based on the category that is the deepest in the hierarchy. We remove any categories
    from `all_categories` that could not be resolved against biolink.

    Example:
    - node has all_categories [biolink:ChemicalEntity, biolink:NamedThing]
    - then node will be assigned biolink:ChemicalEntity as most specific category

    """
    # pre-calculate the mappping table of ID -> most specific category
    mapping_table = (
        nodes.select("id", "all_categories")
        .withColumn("category", F.explode("all_categories"))
        .withColumn(
            "parents", F.udf(get_ancestors_for_category_delimited, T.ArrayType(T.StringType()))(F.col("category"))
        )
        # Our parents list is empty if the parent could not be found, we're removing
        # these elements and ensure there is a non_null check to ensure each element
        # was found in the hierarchy
        .withColumn("depth", F.array_size("parents"))
        .filter(F.col("depth") > 0)
        .withColumn("row_num", F.row_number().over(ps.Window.partitionBy("id").orderBy(F.col("depth").desc())))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
        .select("id", "category")
    )

    return nodes.drop("category").join(mapping_table, on="id", how="left")


def remove_rows_containing_category(
    nodes: ps.DataFrame, categories: List[str], column: str, exclude_sources: Optional[List[str]] = None, **kwargs
) -> ps.DataFrame:
    """Function to remove rows containing a category."""
    if exclude_sources is None:
        exclude_sources = []

    df = nodes.withColumn("_exclude", f.arrays_overlap(f.col("upstream_data_source"), f.lit(exclude_sources))).filter(
        (F.col("_exclude") | ~F.col(column).isin(categories))
    )
    return df.drop("_exclude")
