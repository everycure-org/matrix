"""Filtering functions for the integration pipeline."""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import pyspark as ps
import pyspark.sql.functions as F
import pyspark.sql.functions as f
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def biolink_deduplicate(edges_df: DataFrame, biolink_predicates: DataFrame):
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
        biolink_predicates: JSON object with biolink predicates
    """

    spark = ps.sql.SparkSession.builder.getOrCreate()

    # Load up biolink hierarchy
    biolink_hierarchy = spark.createDataFrame(unnest_biolink_hierarchy(biolink_predicates)).withColumn(
        "predicate", f.concat(f.lit("biolink:"), f.col("predicate"))
    )

    # Enrich edges with path to predicates in biolink hierarchy
    edges_df = edges_df.join(biolink_hierarchy, on="predicate")

    # Compute self join
    res = (
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
        .filter(~f.col("subpath"))
        .select("A.*")
        .drop("parents")
    )

    return res


def remove_rows_containing_category(nodes: DataFrame, categories: List[str], column: str, **kwargs) -> DataFrame:
    """Function to remove rows containing a category."""
    return nodes.filter(~F.col(column).isin(categories))


def unnest_biolink_hierarchy(predicates: List[Dict[str, Any]], parents: Optional[List[str]] = None):
    """Function to unnest biolink predicate hierarchy.

    The biolink predicates are organized in an hierarchical JSON object. To enable
    hierarchical deduplication, the JSON object is pre-processed into a flat pandas
    dataframe that adds the full path to each predicate.

    Args:
        predicates: predicates to unnest
        parents: list of parents in hierarchy
        depth: depth in the hierarchy
    Returns:
        Unnested dataframe
    """

    if parents is None:
        parents = []

    slices = []
    for predicate in predicates:
        name = predicate.get("name")

        # Recurse the children
        if children := predicate.get("children"):
            slices.append(unnest_biolink_hierarchy(children, parents=[*parents, name]))

        slices.append(pd.DataFrame([[name, parents]], columns=["predicate", "parents"]))

    return pd.concat(slices, ignore_index=True)
