import pyspark.sql.types as T

from matrix.utils.pandera_utils import Column, DataFrameSchema

BIOLINK_KG_NODE_SCHEMA = DataFrameSchema(
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

BIOLINK_KG_EDGE_SCHEMA = DataFrameSchema(
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
        "upstream_data_source": Column(T.ArrayType(T.StringType()), nullable=False),
        "num_references": Column(T.IntegerType(), nullable=True),
        "num_sentences": Column(T.IntegerType(), nullable=True),
    },
    unique=["subject", "predicate", "object"],
    strict=True,
)

GT_EDGE_SCHEMA = DataFrameSchema(
    columns={
        "id": Column(T.StringType(), nullable=False),
        "object": Column(T.StringType(), nullable=False),
        "subject": Column(T.StringType(), nullable=False),
        "y": Column(T.IntegerType(), nullable=False),
        "predicate": Column(T.StringType(), nullable=True),
        "flag": Column(T.StringType(), nullable=True),
        "upstream_source": Column(T.StringType(), nullable=False),
        "object_label": Column(T.StringType(), nullable=True),
        "subject_label": Column(T.StringType(), nullable=True),
        "original_object": Column(T.StringType(), nullable=False),
        "original_subject": Column(T.StringType(), nullable=False),
    }
)
