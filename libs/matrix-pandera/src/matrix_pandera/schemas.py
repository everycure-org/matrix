"""
Pandera schema definitions for Matrix node and edge validation.

These schemas are copied from matrix-schema to allow deprecation of that external dependency.
"""

import pandera.pandas as pa
import pyspark.sql.types as T

from matrix_pandera.enums import (
    AgentTypeEnum,
    KnowledgeLevelEnum,
    NodeCategoryEnum,
    PredicateEnum,
)
from matrix_pandera.validator import Column, DataFrameSchema


def get_matrix_node_schema(validate_enumeration_values: bool = True) -> DataFrameSchema:
    """Get the Pandera schema for MatrixNode validation.

    Returns a universal schema that works with both pandas and PySpark DataFrames.
    The actual schema type is determined at validation time based on the DataFrame type.
    """
    if validate_enumeration_values:
        category_checks = [pa.Check.isin([category.value for category in NodeCategoryEnum])]
    else:
        category_checks = []

    return DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "name": Column(T.StringType(), nullable=True),
            "category": Column(T.StringType(), nullable=False, checks=category_checks),
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


def get_matrix_edge_schema(validate_enumeration_values: bool = True) -> DataFrameSchema:
    """Get the Pandera schema for MatrixEdge validation.

    Returns a universal schema that works with both pandas and PySpark DataFrames.
    The actual schema type is determined at validation time based on the DataFrame type.
    """
    if validate_enumeration_values:
        predicate_checks = [pa.Check.isin([predicate.value for predicate in PredicateEnum])]
        knowledge_level_checks = [pa.Check.isin([level.value for level in KnowledgeLevelEnum])]
        agent_type_checks = [pa.Check.isin([agent.value for agent in AgentTypeEnum])]
    else:
        predicate_checks = []
        knowledge_level_checks = []
        agent_type_checks = []

    return DataFrameSchema(
        columns={
            "subject": Column(T.StringType(), nullable=False),
            "predicate": Column(T.StringType(), nullable=False, checks=predicate_checks),
            "object": Column(T.StringType(), nullable=False),
            "knowledge_level": Column(T.StringType(), nullable=True, checks=knowledge_level_checks),
            "agent_type": Column(T.StringType(), nullable=True, checks=agent_type_checks),
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


def get_unioned_node_schema(validate_enumeration_values: bool = True) -> DataFrameSchema:
    """Get the Pandera schema for UnionedNode validation.

    Returns a universal schema that works with both pandas and PySpark DataFrames.
    The actual schema type is determined at validation time based on the DataFrame type.
    """
    if validate_enumeration_values:
        category_checks = [pa.Check.isin([category.value for category in NodeCategoryEnum])]
    else:
        category_checks = []

    return DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "name": Column(T.StringType(), nullable=True),
            "category": Column(T.StringType(), nullable=False, checks=category_checks),
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


def get_unioned_edge_schema(validate_enumeration_values: bool = True) -> DataFrameSchema:
    """Get the Pandera schema for UnionedEdge validation.

    Returns a universal schema that works with both pandas and PySpark DataFrames.
    The actual schema type is determined at validation time based on the DataFrame type.
    """
    if validate_enumeration_values:
        predicate_checks = [pa.Check.isin([predicate.value for predicate in PredicateEnum])]
        knowledge_level_checks = [pa.Check.isin([level.value for level in KnowledgeLevelEnum])]
        agent_type_checks = [pa.Check.isin([agent.value for agent in AgentTypeEnum])]
    else:
        predicate_checks = []
        knowledge_level_checks = []
        agent_type_checks = []

    return DataFrameSchema(
        columns={
            "primary_knowledge_sources": Column(T.ArrayType(T.StringType(), False), nullable=False),
            "subject": Column(T.StringType(), nullable=False),
            "predicate": Column(T.StringType(), nullable=False, checks=predicate_checks),
            "object": Column(T.StringType(), nullable=False),
            "knowledge_level": Column(T.StringType(), nullable=True, checks=knowledge_level_checks),
            "agent_type": Column(T.StringType(), nullable=True, checks=agent_type_checks),
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
