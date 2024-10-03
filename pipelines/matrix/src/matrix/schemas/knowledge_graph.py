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
    """Convenience function that returns the columns of a schema."""
    return list(schema_obj.to_schema().columns.keys())


class KGEdgeSchema(DataFrameModel):
    """Schema for a knowledge graph edges as exposed by the Data API."""

    # fmt: off
    subject:                     StringType()            = pa.Field(nullable = False)
    predicate:                   StringType()            = pa.Field(nullable = False)
    object:                      StringType()            = pa.Field(nullable = False)
    knowledge_level:             StringType()            = pa.Field(nullable = True)
    primary_knowledge_source:    StringType()            = pa.Field(nullable = False)
    aggregator_knowledge_source: ArrayType(StringType()) = pa.Field(nullable = True)
    publications:                ArrayType(StringType()) = pa.Field(nullable = True)
    subject_aspect_qualifier:    StringType()            = pa.Field(nullable = True)
    subject_direction_qualifier: StringType()            = pa.Field(nullable = True)
    object_aspect_qualifier:     StringType()            = pa.Field(nullable = True)
    object_direction_qualifier:  StringType()            = pa.Field(nullable = True)
    # We manually set this for every KG we ingest
    upstream_kg_sources:          ArrayType(StringType()) = pa.Field(nullable = False)
    # fmt: on

    class Config:  # noqa: D106
        coerce = True
        strict = True


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
    upstream_kg_sources:                ArrayType(StringType()) = pa.Field(nullable=False)
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
        return nodes_df.groupBy("id").agg(
            F.first("name").alias("name"),
            F.first("category").alias("category"),
            F.first("description").alias("description"),
            F.collect_set("equivalent_identifiers").alias("equivalent_identifiers"),
            F.collect_set("all_categories").alias("all_categories"),
            F.collect_set("labels").alias("labels"),
            F.collect_set("publications").alias("publications"),
            F.collect_set("international_resource_identifier").alias(
                "international_resource_identifier"
            ),
            F.collect_list("upstream_kg_sources").alias("upstream_kg_sources"),
        )
