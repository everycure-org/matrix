import pandas as pd

from typing import Dict, Any, List, Optional
from pyspark.sql import DataFrame

import pyspark.sql.functions as f


def unnest(predicates: List[Dict[str, Any]], parents: Optional[List[str]] = None):
    """
    Function to unnest biolink predicates.

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
            slices.append(unnest(children, parents=[*parents, name]))

        slices.append(pd.DataFrame([[name, parents]], columns=["predicate", "parents"]))

    return pd.concat(slices, ignore_index=True)


def biolink_deduplicate(edges_df: DataFrame, biolink_hierarchy: DataFrame):
    # Filter hierarical edges
    edges_df = edges_df.join(
        biolink_hierarchy.withColumn("predicate", f.concat(f.lit("biolink:"), f.col("predicate"))), on="predicate"
    )

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
