"""This module contains the schema for the knowledge graph exposed by the Data API.

It is defined in Pandera which is strongly inspired by Pydantic.
"""

import pandera.pyspark as pa
from pandera.pyspark import DataFrameModel
import pyspark.sql.types as T


class KGEdgeSchema(DataFrameModel):
    """Schema for a knowledge graph edges as exposed by the Data API."""

    subject: T.StringType() = pa.Field(nullable=False)
    predicate: T.StringType() = pa.Field(nullable=False)
    object: T.StringType() = pa.Field(nullable=False)
    knowledge_level: T.StringType() = pa.Field(nullable=False)
    primary_knowledge_source: T.StringType() = pa.Field(nullable=False)
    aggregator_knowledge_source: T.ArrayType(T.StringType()) = pa.Field(nullable=True)
    publications: T.ArrayType(T.StringType()) = pa.Field(nullable=True)
    subject_aspect_qualifier: T.StringType() = pa.Field(nullable=True)
    subject_direction_qualifier: T.StringType() = pa.Field(nullable=True)
    object_aspect_qualifier: T.StringType() = pa.Field(nullable=True)
    object_direction_qualifier: T.StringType() = pa.Field(nullable=True)

    # We manually set this for every KG we ingest
    upstream_kg_source: T.ArrayType(T.StringType()) = pa.Field(nullable=False)

    class Config:  # noqa: D106
        coerce = True
        strict = True


class KGNodeSchema(DataFrameModel):
    """Schema for a knowledge graph nodes as exposed by the Data API."""

    id: T.StringType() = pa.Field(nullable=False)
    name: T.StringType() = pa.Field(nullable=False)
    category: T.StringType() = pa.Field(nullable=False)
    description: T.StringType() = pa.Field(nullable=True)

    equivalent_identifiers: T.ArrayType(T.StringType()) = pa.Field(nullable=True)
    all_categories: T.ArrayType(T.StringType()) = pa.Field(nullable=True)
    publications: T.ArrayType(T.StringType()) = pa.Field(nullable=True)
    label: T.StringType() = pa.Field(nullable=True)
    international_resource_identifier: T.StringType() = pa.Field(nullable=True)

    # We manually set this for every KG we ingest
    upstream_kg_source: T.ArrayType(T.StringType()) = pa.Field(nullable=False)

    class Config:  # noqa: D106
        coerce = True
        strict = True
