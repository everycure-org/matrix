import pandas as pd
import pandera.pyspark as pa
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from matrix.pipelines.integration.filters import determine_most_specific_category
from matrix.schemas.knowledge_graph import KGEdgeSchema, cols_for_schema, KGNodeSchema

# FUTURE: We should likely not need to rename these columns as we do below
# However, KGX is currently not as performant as we need it to be thus
# we do it manually with spark. This ought to be improved, e.g. by
# adding parquet support to KGX.
# or implementing a custom KGX version that leverages spark for higher performance
# https://github.com/everycure-org/matrix/issues/474

ROBOKOP_SEPARATOR = "\x1f"


@pa.check_output(KGNodeSchema)
def transform_robo_nodes(nodes_df: DataFrame, biolink_categories_df: pd.DataFrame) -> DataFrame:
    """Transform Robokop nodes to our target schema.

    Args:
        nodes_df: Nodes DataFrame.
        biolink_categories_df: Biolink categories DataFrame.

    Returns:
        Transformed DataFrame.
    """
    # NOTE: This function was partially generated using AI assistance.
    # fmt: off


    return (
        nodes_df
        .withColumn("upstream_data_source",              F.array(F.lit("robokop")))
        .withColumn("all_categories",                    F.split(F.col("category:LABEL"), ROBOKOP_SEPARATOR))
        .withColumn("equivalent_identifiers",            F.split(F.col("equivalent_identifiers:string[]"), ROBOKOP_SEPARATOR))
        .withColumn("labels",                            F.array(F.col("all_categories")))
        .withColumn("publications",                      F.array(F.lit(None)))
        .withColumn("international_resource_identifier", F.lit(None))
        .withColumnRenamed("id:ID", "id")
        .withColumnRenamed("name:string", "name")
        .withColumnRenamed("description:string", "description")
        # getting most specific category
        .transform(determine_most_specific_category, biolink_categories_df)
        .select(*cols_for_schema(KGNodeSchema))
    )
    # fmt: on


@pa.check_output(KGEdgeSchema)
def transform_robo_edges(edges_df: DataFrame) -> DataFrame:
    """Transform Robokop edges to our target schema.

    Args:
        edges_df: Edges DataFrame.

    Returns:
        Transformed DataFrame.
    """
    # NOTE: This function was partially generated using AI assistance.
    # fmt: off
    return (
        edges_df
        .withColumnRenamed("subject:START_ID",                  "subject")
        .withColumnRenamed("predicate:TYPE",                    "predicate")
        .withColumnRenamed("object:END_ID",                     "object")
        .withColumnRenamed("knowledge_level:string",            "knowledge_level")
        .withColumnRenamed("primary_knowledge_source:string",   "primary_knowledge_source")
        .withColumnRenamed("object_aspect_qualifier:string",    "object_aspect_qualifier")
        .withColumnRenamed("object_direction_qualifier:string", "object_direction_qualifier")
        .withColumn("upstream_data_source",                     F.array(F.lit("robokop")))
        .withColumn("publications",                             F.split(F.col("publications:string[]"), ROBOKOP_SEPARATOR))
        .withColumn("aggregator_knowledge_source",              F.split(F.col("aggregator_knowledge_source:string[]"), ROBOKOP_SEPARATOR))
        .withColumn("subject_aspect_qualifier",                 F.lit(None).cast(T.StringType()))
        .withColumn("subject_direction_qualifier",              F.lit(None).cast(T.StringType()))
        # final selection of columns
        .select(*cols_for_schema(KGEdgeSchema))
    )
    # fmt: on
