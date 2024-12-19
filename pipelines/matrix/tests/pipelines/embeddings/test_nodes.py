import numpy as np
import pytest
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StringType,
    ArrayType,
    DoubleType,
    StructType,
    StructField,
    FloatType,
)
from matrix.pipelines.embeddings.nodes import extract_topological_embeddings


@pytest.fixture
def sample_nodes_df(spark):
    """Create sample nodes dataframe."""
    schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("name", StringType(), True),
            StructField("category", StringType(), True),
        ]
    )

    data = [("node1", "Node 1", "Category A"), ("node2", "Node 2", "Category B"), ("node3", "Node 3", "Category A")]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_string_embeddings_df(spark):
    """Create sample embeddings dataframe with string embeddings."""
    schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("topological_embedding", StringType(), True),
            StructField("pca_embedding", ArrayType(FloatType()), True),
        ]
    )

    data = [
        ("node1", "[1.0, 2.0, 3.0]", [0.1, 0.2]),
        ("node2", "[4.0, 5.0, 6.0]", [0.3, 0.4]),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_array_embeddings_df(spark):
    """Create sample embeddings dataframe with array embeddings."""
    schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("topological_embedding", ArrayType(DoubleType()), True),
            StructField("pca_embedding", ArrayType(DoubleType()), True),
        ]
    )

    data = [
        ("node1", [1.0, 2.0, 3.0], [0.1, 0.2]),
        ("node2", [4.0, 5.0, 6.0], [0.3, 0.4]),
    ]

    return spark.createDataFrame(data, schema)


def test_extract_topological_embeddings_string(sample_nodes_df, sample_string_embeddings_df):
    """Test extraction of topological embeddings when stored as strings."""
    result = extract_topological_embeddings(sample_string_embeddings_df, sample_nodes_df, "topological_embedding")

    # Check schema
    assert "topological_embedding" in result.columns
    assert "pca_embedding" in result.columns
    assert isinstance(result.schema["topological_embedding"].dataType, ArrayType)

    # Check data
    result_data = result.collect()
    assert len(result_data) == 3  # All nodes should be present (left join)

    # Check specific values
    node1 = result.filter(F.col("id") == "node1").first()
    # check if almost equal
    assert np.allclose(node1.topological_embedding, [1.0, 2.0, 3.0])
    assert np.allclose(node1.pca_embedding, [0.1, 0.2])


def test_extract_topological_embeddings_array(sample_nodes_df, sample_array_embeddings_df):
    """Test extraction of topological embeddings when stored as arrays."""
    result = extract_topological_embeddings(sample_array_embeddings_df, sample_nodes_df, "topological_embedding")

    # Check schema
    assert "topological_embedding" in result.columns
    assert "pca_embedding" in result.columns
    assert isinstance(result.schema["topological_embedding"].dataType, ArrayType)

    # Check data
    result_data = result.collect()
    assert len(result_data) == 3  # All nodes should be present (left join)

    # Check specific values
    node1 = result.filter(F.col("id") == "node1").first()
    # check if almost equal
    assert np.allclose(node1.topological_embedding, [1.0, 2.0, 3.0])
    assert np.allclose(node1.pca_embedding, [0.1, 0.2])


def test_extract_topological_embeddings_missing_nodes(sample_nodes_df, sample_array_embeddings_df):
    """Test handling of nodes without embeddings."""
    result = extract_topological_embeddings(sample_array_embeddings_df, sample_nodes_df, "topological_embedding")

    # Check node without embedding
    node3 = result.filter(F.col("id") == "node3").first()
    assert node3.topological_embedding is None
    assert node3.pca_embedding is None
