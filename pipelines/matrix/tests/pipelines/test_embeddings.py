import numpy as np
import pyspark.sql as ps
import pytest
from matrix.pipelines.embeddings.nodes import (
    _cast_to_array,
    extract_topological_embeddings,
    ingest_nodes,
    reduce_embeddings_dimension,
)
from pyspark.ml.feature import PCA
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def sample_input_df(spark: ps.SparkSession) -> ps.DataFrame:
    data = [
        {
            "id": "1",
            "name": "Test Node",
            "category": "TestCategory",
            "description": "Test Description",
            "upstream_data_source": ["source1", "source2"],
        },
        {
            "id": "2",
            "name": "Test Node 2",
            "category": "TestCategory2",
            "description": "Test Description 2",
            "upstream_data_source": ["source3"],
        },
    ]
    return spark.createDataFrame(data)


@pytest.fixture
def sample_embeddings_df(spark: ps.SparkSession) -> ps.DataFrame:
    """Create a sample dataframe with embeddings."""
    schema = ps.types.StructType(
        [
            ps.types.StructField("id", ps.types.StringType(), False),
            ps.types.StructField("embedding", ps.types.ArrayType(ps.types.FloatType(), True), False),
        ]
    )

    data = [
        {"id": "1", "embedding": [1.0, 2.0, 3.0, 4.0]},
        {"id": "2", "embedding": [2.0, 3.0, 4.0, 5.0]},
        {"id": "3", "embedding": [3.0, 4.0, 5.0, 6.0]},
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def pca_transformer() -> PCA:
    """Create a PCA transformer."""
    return PCA(k=2)  # Reduce to 2 dimensions


@pytest.fixture
def sample_nodes_df(spark):
    """Create sample nodes dataframe."""
    schema = ps.types.StructType(
        [
            ps.types.StructField("id", ps.types.StringType(), False),
            ps.types.StructField("name", ps.types.StringType(), True),
            ps.types.StructField("category", ps.types.StringType(), True),
        ]
    )

    data = [("node1", "Node 1", "Category A"), ("node2", "Node 2", "Category B"), ("node3", "Node 3", "Category A")]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_string_embeddings_df(spark):
    """Create sample embeddings dataframe with string embeddings."""
    schema = ps.types.StructType(
        [
            ps.types.StructField("id", ps.types.StringType(), False),
            ps.types.StructField("topological_embedding", ps.types.StringType(), True),
            ps.types.StructField("pca_embedding", ps.types.ArrayType(ps.types.FloatType()), True),
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
    schema = ps.types.StructType(
        [
            ps.types.StructField("id", ps.types.StringType(), False),
            ps.types.StructField("topological_embedding", ps.types.ArrayType(ps.types.DoubleType()), True),
            ps.types.StructField("pca_embedding", ps.types.ArrayType(ps.types.DoubleType()), True),
        ]
    )

    data = [
        ("node1", [1.0, 2.0, 3.0], [0.1, 0.2]),
        ("node2", [4.0, 5.0, 6.0], [0.3, 0.4]),
    ]

    return spark.createDataFrame(data, schema)


def test_ingest_nodes_basic(spark: ps.SparkSession, sample_input_df: ps.DataFrame) -> None:
    """Test basic functionality of ingest_nodes."""
    result = ingest_nodes(sample_input_df)

    expected_data = [
        {
            "label": "TestCategory",
            "property_keys": ["name", "category", "description"],
            "property_values": ["Test Node", "TestCategory", "Test Description"],
            "array_property_keys": ["upstream_data_source"],
            "array_property_values": [["source1", "source2"]],
        },
        {
            "label": "TestCategory2",
            "property_keys": ["name", "category", "description"],
            "property_values": ["Test Node 2", "TestCategory2", "Test Description 2"],
            "array_property_keys": ["upstream_data_source"],
            "array_property_values": [["source3"]],
        },
    ]

    expected_df = spark.createDataFrame(expected_data)

    # Compare only the columns we care about
    result_subset = result.select(*expected_df.columns)
    assertDataFrameEqual(result_subset, expected_df)


def test_ingest_nodes_empty_df(spark: ps.SparkSession) -> None:
    """Test handling of empty dataframe."""
    empty_df = spark.createDataFrame(
        [], "id string, name string, category string, description string, upstream_data_source array<string>"
    )

    result = ingest_nodes(empty_df)
    assert result.count() == 0


def test_reduce_embeddings_dimension_with_transformation(
    sample_embeddings_df: ps.DataFrame, pca_transformer: PCA
) -> None:
    """Test dimensionality reduction when skip=False."""
    # Arrange
    params = {"transformer": pca_transformer, "input": "embedding", "output": "pca_embedding", "skip": False}

    # Act
    result_df = reduce_embeddings_dimension(sample_embeddings_df, **params)

    # Assert
    # Check schema
    assert "pca_embedding" in result_df.columns

    # Check output dimensions
    result_rows = result_df.collect()
    assert len(result_rows) == 3  # Same number of rows
    assert len(result_rows[0]["pca_embedding"]) == 2  # Reduced to 2 dimensions

    # Check data type
    assert result_df.schema["pca_embedding"].dataType.typeName() == "array"
    assert result_df.schema["pca_embedding"].dataType.elementType.typeName() == "float"


def test_reduce_embeddings_dimension_skip(sample_embeddings_df: ps.DataFrame, pca_transformer: PCA) -> None:
    """Test when skip=True, should return original embeddings."""
    # Arrange
    params = {"transformer": pca_transformer, "input": "embedding", "output": "pca_embedding", "skip": True}

    # Act
    result_df = reduce_embeddings_dimension(sample_embeddings_df, **params)

    # Assert
    # Check schema
    assert "pca_embedding" in result_df.columns

    # Check dimensions remain unchanged
    result_rows = result_df.collect()
    assert len(result_rows) == 3
    assert len(result_rows[0]["pca_embedding"]) == 4  # Original dimension

    # Check values are preserved
    original_values = sample_embeddings_df.collect()
    for orig, result in zip(original_values, result_rows):
        assert np.array_equal(orig["embedding"], result["pca_embedding"])


def test_reduce_embeddings_dimension_invalid_input(sample_embeddings_df: ps.DataFrame, pca_transformer: PCA) -> None:
    """Test with invalid input column name."""
    # Arrange
    params = {"transformer": pca_transformer, "input": "nonexistent_column", "output": "pca_embedding", "skip": False}

    # Act & Assert
    with pytest.raises(Exception):  # Should raise an exception for invalid column
        reduce_embeddings_dimension(sample_embeddings_df, **params)


def test_cast_to_array(sample_string_embeddings_df):
    """Test extraction of topological embeddings when stored as strings."""
    result = _cast_to_array(sample_string_embeddings_df, "topological_embedding")

    # Check schema
    assert "topological_embedding" in result.columns
    assert isinstance(result.schema["topological_embedding"].dataType, ps.types.ArrayType)

    # Check specific values
    node1 = result.filter(ps.functions.col("id") == "node1").first()
    # check if almost equal
    assert np.allclose(node1.topological_embedding, [1.0, 2.0, 3.0])
    assert np.allclose(node1.pca_embedding, [0.1, 0.2])
