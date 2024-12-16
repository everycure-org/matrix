from typing import List

import pandera
import pyspark.sql.functions as F
import pyspark.sql.types as T

from pandera.pyspark import DataFrameModel
from pyspark.sql import DataFrame


def cols_for_schema(schema_obj: DataFrameModel) -> List[str]:
    """Convenience function that returns the columns of a schema.

    The function returns all the columns of the passed model. This is convenient for
    selecting the columns of a schema in a pipeline using pyspark which then drops all
    other columns.
    """
    return list(schema_obj.to_schema().columns.keys())


class KGEdgeSchema(DataFrameModel):
    """Schema for a knowledge graph edges as exposed by the Data API."""

    # fmt: off
    subject:                     T.StringType()            = pandera.pyspark.Field(nullable = False) # type: ignore
    predicate:                   T.StringType()            = pandera.pyspark.Field(nullable = False) # type: ignore
    object:                      T.StringType()            = pandera.pyspark.Field(nullable = False) # type: ignore
    knowledge_level:             T.StringType()            = pandera.pyspark.Field(nullable = True) # type: ignore
    primary_knowledge_source:    T.StringType()            = pandera.pyspark.Field(nullable = True) # type: ignore
    aggregator_knowledge_source: T.ArrayType(T.StringType()) = pandera.pyspark.Field(nullable = True) # type: ignore
    publications:                T.ArrayType(T.StringType()) = pandera.pyspark.Field(nullable = True) # type: ignore
    subject_aspect_qualifier:    T.StringType()            = pandera.pyspark.Field(nullable = True) # type: ignore
    subject_direction_qualifier: T.StringType()            = pandera.pyspark.Field(nullable = True) # type: ignore
    object_aspect_qualifier:     T.StringType()            = pandera.pyspark.Field(nullable = True) # type: ignore
    object_direction_qualifier:  T.StringType()            = pandera.pyspark.Field(nullable = True) # type: ignore
    # We manually set this for every KG we ingest
    upstream_data_source:          T.ArrayType(T.StringType()) = pandera.pyspark.Field(nullable = False) # type: ignore
    # fmt: on

    class Config:  # noqa: D106
        coerce = True
        strict = False

    @classmethod
    def group_edges_by_id(cls, concatenated_edges_df: DataFrame) -> DataFrame:
        return (
            concatenated_edges_df.groupBy(["subject", "predicate", "object"])
            .agg(
                F.flatten(F.collect_set("upstream_data_source")).alias("upstream_data_source"),
                # TODO: we shouldn't just take the first one but collect these values from multiple upstream sources
                F.first("knowledge_level").alias("knowledge_level"),
                F.first("subject_aspect_qualifier").alias("subject_aspect_qualifier"),
                F.first("subject_direction_qualifier").alias("subject_direction_qualifier"),
                F.first("object_direction_qualifier").alias("object_direction_qualifier"),
                F.first("object_aspect_qualifier").alias("object_aspect_qualifier"),
                F.first("primary_knowledge_source").alias("primary_knowledge_source"),
                F.flatten(F.collect_set("aggregator_knowledge_source")).alias("aggregator_knowledge_source"),
                F.flatten(F.collect_set("publications")).alias("publications"),
            )
            .select(*cols_for_schema(KGEdgeSchema))
        )


class KGNodeSchema(DataFrameModel):
    """Schema for a knowledge graph nodes as exposed by the Data API."""

    # fmt: off
    id:                                T.StringType()            = pandera.pyspark.Field(nullable=False) # type: ignore
    name:                              T.StringType()            = pandera.pyspark.Field(nullable=True) # type: ignore #TODO should this be nullable?
    category:                          T.StringType()            = pandera.pyspark.Field(nullable=False) # type: ignore
    description:                       T.StringType()            = pandera.pyspark.Field(nullable=True) # type: ignore
    equivalent_identifiers:            T.ArrayType(T.StringType()) = pandera.pyspark.Field(nullable=True) # type: ignore
    all_categories:                    T.ArrayType(T.StringType()) = pandera.pyspark.Field(nullable=True) # type: ignore
    publications:                      T.ArrayType(T.StringType()) = pandera.pyspark.Field(nullable=True) # type: ignore
    labels:                            T.ArrayType(T.StringType()) = pandera.pyspark.Field(nullable=True) # type: ignore
    international_resource_identifier: T.StringType()            = pandera.pyspark.Field(nullable=True) # type: ignore
    # We manually set this for every KG we ingest
    upstream_data_source:                T.ArrayType(T.StringType()) = pandera.pyspark.Field(nullable=False) # type: ignore
    #upstream_kg_node_ids:                MapType(StringType(), StringType()) = pa.Field(nullable=True)
    # fmt: on

    class Config:  # noqa: D106
        coerce = True
        strict = True

    @classmethod
    def group_nodes_by_id(cls, nodes_df: DataFrame) -> DataFrame:
        """Utility function to group nodes by id.

        This should be used after the IDs are normalized so we can combine node properties from
        multiple upstream KGs.

        """
        # FUTURE: We should improve selection of name and description currently
        # selecting the first non-null, which might not be as desired.
        # fmt: off
        return (
            nodes_df.groupBy("id")
            .agg(
                F.first("name").alias("name"),
                F.first("category").alias("category"),
                F.first("description").alias("description"),
                F.first("international_resource_identifier").alias("international_resource_identifier"),
                F.flatten(F.collect_set("equivalent_identifiers")).alias("equivalent_identifiers"),
                F.flatten(F.collect_set("all_categories")).alias("all_categories"),
                F.flatten(F.collect_set("labels")).alias("labels"),
                F.flatten(F.collect_set("publications")).alias("publications"),
                F.flatten(F.collect_set("upstream_data_source")).alias("upstream_data_source"),
            )
            .select(*cols_for_schema(KGNodeSchema))
        )
