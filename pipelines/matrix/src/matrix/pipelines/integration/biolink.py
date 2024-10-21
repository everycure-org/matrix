import pandas as pd

from typing import Dict, Any, List, Optional
from pyspark.sql import DataFrame

import pyspark.sql.functions as f
import pyspark as ps


def _unnest(predicates: List[Dict[str, Any]], parents: Optional[List[str]] = None):
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
            slices.append(_unnest(children, parents=[*parents, name]))

        slices.append(pd.DataFrame([[name, parents]], columns=["predicate", "parents"]))

    return pd.concat(slices, ignore_index=True)


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

    biolink_hierarchy = spark.createDataFrame(_unnest(biolink_predicates)).withColumn(
        "predicate", f.concat(f.lit("biolink:"), f.col("predicate"))
    )

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


def filter_semmed(
    edges_df: DataFrame,
    pubmed_mapping: pd.DataFrame,
    num_pairs: float = 3.7e7 * 20,
    publication_threshold: int = 1,
    ndg_threshold: float = 0.6,
) -> DataFrame:
    spark = ps.sql.SparkSession.builder.getOrCreate()

    pubmed_mapping_spark = (
        spark.createDataFrame(pubmed_mapping)
        .withColumn("pmids", f.array_distinct(f.col("pmids")))
        .withColumn("num_pmids", f.array_size(f.col("pmids")))
    )

    # NOTE: Let's think of what features we expose as part of feature store
    # and where we apply filtering
    return (
        edges_df.withColumn("num_publications", f.size(f.col("publications")))
        .join(
            pubmed_mapping_spark.withColumnRenamed("curie", "subject")
            .withColumnRenamed("pmids", "subject_pmids")
            .withColumnRenamed("num_pmids", "num_subject_pmids"),
            on="subject",
            how="left",
        )
        .join(
            pubmed_mapping_spark.withColumnRenamed("curie", "object")
            .withColumnRenamed("pmids", "object_pmids")
            .withColumnRenamed("num_pmids", "num_object_pmids"),
            on="object",
            how="left",
        )
        .withColumn("num_common_pmids", f.array_size(f.array_intersect(f.col("subject_pmids"), f.col("object_pmids"))))
        .withColumn(
            "ndg",
            (
                f.max(f.log2(f.col("num_subject_pmids")), f.log2(f.col("num_object_pmids")))
                - f.log2(f.col("num_common_pmids"))
            )
            / (f.log2(f.lit(num_pairs)) - f.min(f.log2(f.col("num_subject_pmids")), f.log2(f.col("num_object_pmids")))),
        )
        .filter(f.col("ndg") < f.lit(ndg_threshold))
        .filter(f.col("num_publications") > f.lit(publication_threshold))
    )
