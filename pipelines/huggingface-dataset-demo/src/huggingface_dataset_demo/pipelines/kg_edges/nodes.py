from __future__ import annotations

from pyspark.sql import (
    DataFrame,  # type: ignore[import-not-found]
    SparkSession,  # type: ignore[import-not-found]
)
from pyspark.sql import functions as F  # type: ignore[import-not-found]


def build_kg_edges_spark() -> DataFrame:
    """
    Build a 100k-row Spark DataFrame with columns:
    subject, predicate, object, description.
    """
    spark = SparkSession.builder.getOrCreate()

    base = spark.range(0, 100_000).withColumnRenamed("id", "idx")

    # A small set of example predicates (can be expanded later)
    predicates = [
        "biolink:related_to",
        "biolink:treats",
        "biolink:causes",
        "biolink:interacts_with",
        "biolink:associated_with",
        "biolink:contraindicated_for",
        "biolink:ameliorates",
        "biolink:exacerbates",
        "biolink:biomarker_for",
        "biolink:targets",
    ]

    pred_expr = F.expr(
        "case "
        + " ".join(f"when (idx % {len(predicates)}) = {i} then '{p}'" for i, p in enumerate(predicates))
        + f" else '{predicates[0]}' end"
    )

    df = base.select(
        F.concat(F.lit("EC-DRUG-"), F.col("idx")).alias("subject"),
        pred_expr.alias("predicate"),
        F.concat(F.lit("EC-DISEASE-"), (F.col("idx") % F.lit(1000))).alias("object"),
        F.concat(F.lit("Edge "), F.col("idx")).alias("description"),
    )

    return df


def spark_to_jsonl(df: DataFrame):
    """
    Convert a Spark DataFrame to pandas for JSONL writing via Kedro JSONDataset.
    """
    return df.toPandas()
