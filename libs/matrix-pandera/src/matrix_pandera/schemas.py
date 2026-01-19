"""Pandera schema definitions for Matrix node and edge validation."""

import pyspark.sql.types as T

from matrix_pandera.validator import Column, DataFrameSchema


def get_matrix_node_schema() -> DataFrameSchema:
    """Get the Pandera schema for MatrixNode validation.

    Returns a universal schema that works with both pandas and PySpark DataFrames.
    The actual schema type is determined at validation time based on the DataFrame type.
    """
    return DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "name": Column(T.StringType(), nullable=True),
            "category": Column(T.StringType(), nullable=False),
            "description": Column(T.StringType(), nullable=True),
            "equivalent_identifiers": Column(T.ArrayType(T.StringType()), nullable=True),
            "all_categories": Column(T.ArrayType(T.StringType()), nullable=True),
            "publications": Column(T.ArrayType(T.StringType()), nullable=True),
            "labels": Column(T.ArrayType(T.StringType()), nullable=True),
            "international_resource_identifier": Column(T.StringType(), nullable=True),
            "upstream_data_source": Column(T.ArrayType(T.StringType()), nullable=True),
        },
        unique=["id"],
        strict=True,
    )


def get_matrix_edge_schema() -> DataFrameSchema:
    """Get the Pandera schema for MatrixEdge validation.

    Returns a universal schema that works with both pandas and PySpark DataFrames.
    The actual schema type is determined at validation time based on the DataFrame type.
    """
    return DataFrameSchema(
        columns={
            "subject": Column(T.StringType(), nullable=False),
            "predicate": Column(T.StringType(), nullable=False),
            "object": Column(T.StringType(), nullable=False),
            "knowledge_level": Column(T.StringType(), nullable=True),
            "agent_type": Column(T.StringType(), nullable=True),
            "primary_knowledge_source": Column(T.StringType(), nullable=True),
            "aggregator_knowledge_source": Column(T.ArrayType(T.StringType()), nullable=True),
            "publications": Column(T.ArrayType(T.StringType()), nullable=True),
            "subject_aspect_qualifier": Column(T.StringType(), nullable=True),
            "subject_direction_qualifier": Column(T.StringType(), nullable=True),
            "object_aspect_qualifier": Column(T.StringType(), nullable=True),
            "object_direction_qualifier": Column(T.StringType(), nullable=True),
            "upstream_data_source": Column(T.ArrayType(T.StringType()), nullable=True),
            "num_references": Column(T.IntegerType(), nullable=True),
            "num_sentences": Column(T.IntegerType(), nullable=True),
        },
        unique=["subject", "predicate", "object"],
        strict=True,
    )


def get_unioned_node_schema() -> DataFrameSchema:
    """Get the Pandera schema for UnionedNode validation.

    Returns a universal schema that works with both pandas and PySpark DataFrames.
    The actual schema type is determined at validation time based on the DataFrame type.
    """
    return DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "name": Column(T.StringType(), nullable=True),
            "category": Column(T.StringType(), nullable=False),
            "description": Column(T.StringType(), nullable=True),
            "equivalent_identifiers": Column(T.ArrayType(T.StringType()), nullable=True),
            "all_categories": Column(T.ArrayType(T.StringType()), nullable=True),
            "publications": Column(T.ArrayType(T.StringType()), nullable=True),
            "labels": Column(T.ArrayType(T.StringType()), nullable=True),
            "international_resource_identifier": Column(T.StringType(), nullable=True),
            "upstream_data_source": Column(T.ArrayType(T.StringType()), nullable=True),
        },
        unique=["id"],
        strict=True,
    )


def get_unioned_edge_schema() -> DataFrameSchema:
    """Get the Pandera schema for UnionedEdge validation.

    Returns a universal schema that works with both pandas and PySpark DataFrames.
    The actual schema type is determined at validation time based on the DataFrame type.
    """
    return DataFrameSchema(
        columns={
            "primary_knowledge_sources": Column(T.ArrayType(T.StringType(), False), nullable=False),
            "subject": Column(T.StringType(), nullable=False),
            "predicate": Column(T.StringType(), nullable=False),
            "object": Column(T.StringType(), nullable=False),
            "knowledge_level": Column(T.StringType(), nullable=True),
            "agent_type": Column(T.StringType(), nullable=True),
            "primary_knowledge_source": Column(T.StringType(), nullable=True),
            "aggregator_knowledge_source": Column(T.ArrayType(T.StringType()), nullable=True),
            "publications": Column(T.ArrayType(T.StringType()), nullable=True),
            "subject_aspect_qualifier": Column(T.StringType(), nullable=True),
            "subject_direction_qualifier": Column(T.StringType(), nullable=True),
            "object_aspect_qualifier": Column(T.StringType(), nullable=True),
            "object_direction_qualifier": Column(T.StringType(), nullable=True),
            "upstream_data_source": Column(T.ArrayType(T.StringType()), nullable=True),
            "num_references": Column(T.IntegerType(), nullable=True),
            "num_sentences": Column(T.IntegerType(), nullable=True),
        },
        unique=["subject", "predicate", "object"],
        strict=True,
    )
