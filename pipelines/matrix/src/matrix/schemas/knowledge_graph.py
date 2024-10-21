"""This module contains the schema for the knowledge graph exposed by the Data API.

It is defined in Pandera which is strongly inspired by Pydantic.
"""

from typing import List

import pandera.pyspark as pa
import pyspark.sql.functions as F
from pandera.pyspark import DataFrameModel
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StringType


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
    subject:                     StringType()            = pa.Field(nullable = False)
    predicate:                   StringType()            = pa.Field(nullable = False)
    object:                      StringType()            = pa.Field(nullable = False)
    knowledge_level:             StringType()            = pa.Field(nullable = True)
    primary_knowledge_source:    ArrayType(StringType()) = pa.Field(nullable = True)
    aggregator_knowledge_source: ArrayType(StringType()) = pa.Field(nullable = True)
    publications:                ArrayType(StringType()) = pa.Field(nullable = True)
    subject_aspect_qualifier:    StringType()            = pa.Field(nullable = True)
    subject_direction_qualifier: StringType()            = pa.Field(nullable = True)
    object_aspect_qualifier:     StringType()            = pa.Field(nullable = True)
    object_direction_qualifier:  StringType()            = pa.Field(nullable = True)
    # We manually set this for every KG we ingest
    upstream_data_source:          ArrayType(StringType()) = pa.Field(nullable = False)
    # fmt: on

    class Config:  # noqa: D106
        coerce = True
        strict = True

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
    id:                                StringType()            = pa.Field(nullable=False)
    name:                              StringType()            = pa.Field(nullable=True) #TODO should this be nullable?
    category:                          StringType()            = pa.Field(nullable=False)
    description:                       StringType()            = pa.Field(nullable=True)
    equivalent_identifiers:            ArrayType(StringType()) = pa.Field(nullable=True)
    all_categories:                    ArrayType(StringType()) = pa.Field(nullable=True)
    publications:                      ArrayType(StringType()) = pa.Field(nullable=True)
    labels:                            ArrayType(StringType()) = pa.Field(nullable=True)
    international_resource_identifier: StringType()            = pa.Field(nullable=True)
    # We manually set this for every KG we ingest
    upstream_data_source:                ArrayType(StringType()) = pa.Field(nullable=False)
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
                # keep all upstream node IDs as a map {kg_name: node_id} so we can trace them later
                # TODO finish
                # F.create_map("upstream_data_source", "original_id").alias("upstream_kg_node_ids")
            )
            .select(*cols_for_schema(KGNodeSchema))
        )
