"""This module contains the schema for the knowledge graph exposed by the Data API.

It is defined in Pandera which is strongly inspired by Pydantic.
"""

import pandera.pyspark as pa
from pandera.pyspark import DataFrameModel
from pyspark.sql.types import *
from typing import List


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
    upstream_kg_source:          ArrayType(StringType()) = pa.Field(nullable = False)
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
    label:                             StringType()            = pa.Field(nullable=True)
    international_resource_identifier: StringType()            = pa.Field(nullable=True)
    # We manually set this for every KG we ingest
    upstream_kg_source:                ArrayType(StringType()) = pa.Field(nullable=False)
    # fmt: on

    class Config:  # noqa: D106
        coerce = True
        strict = True
