from random import choice
from typing import Iterable, Iterator, Sequence, Tuple, Type, TypeVar
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pyspark.sql as ps
import pyspark.sql.types as ty
import pytest
from matrix.pipelines.embeddings import nodes
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


@pytest.fixture
def pre_embedding(spark: ps.SparkSession) -> ps.DataFrame:
    """Provides the outcome of the integration."""

    return spark.createDataFrame(
        [
            (1, "a", "A"),
            (2, "a", "B"),
            (3, "a", "C"),
            (4, "a", "D"),
        ],
        schema=("id", "name", "category"),
    )


@pytest.fixture
def scope():
    return "foo"


@pytest.fixture
def model():
    return "bar"


@pytest.fixture
def embeddings_cache(scope: str, model: str, spark: ps.SparkSession) -> ps.DataFrame:
    return spark.createDataFrame(
        [
            (scope, model, "aA", [1]),
            (scope, model, "cC", [1]),
        ],
        schema=("scope", "model", "key", "value"),
    )


U = TypeVar("U")
V = TypeVar("V")


def return_constant(docs: Iterable[U]) -> Iterator[Tuple[U, V]]:
    for doc in docs:
        yield doc, [1.0] * choice((1, 2))


@pytest.fixture
def mock_encoder() -> Mock:
    encoder = Mock()

    encoder.embed = Mock(side_effect=return_constant)
    return encoder


@pytest.fixture
def mock_encoder2(mock_encoder) -> Mock:
    return mock_encoder


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


def test_cached_embeddings_can_get_loaded(pre_embedding: ps.DataFrame):
    scope, model = "foo", "bar"
    dummy_cache = pre_embedding.sparkSession.createDataFrame(
        [
            (scope, model, "aA", [1.0]),
            (scope, model, "bB", [2.0]),
        ],
        schema=ty.StructType(
            [
                ty.StructField("scope", ty.StringType(), nullable=False),
                ty.StructField("model", ty.StringType(), nullable=False),
                ty.StructField("id", ty.StringType(), nullable=False),
                ty.StructField("embedding", ty.ArrayType(ty.DoubleType()), nullable=False),
            ]
        ),
    )

    result = nodes.load_embeddings_from_cache(
        dataframe=pre_embedding,
        cache=dummy_cache,
        model=model,
        scope=scope,
    )

    expected = result.sparkSession.createDataFrame(
        [(1, "a", "A", scope, model, [1.0]), (2, "a", "B", scope, model, [2.0])],
        schema=("id", "name", "category", "scope", "model", "embedding"),
    )

    expected.show()
    result.show()
    assert sorted(expected.columns) == sorted(result.columns)  # order of columns is unimportant
    assertDataFrameEqual(result.select(expected.columns), expected)


def test_re_embedding_is_a_noop(
    pre_embedding: ps.DataFrame,
    embeddings_cache: ps.DataFrame,
    scope: str,
    model: str,
):
    """Given a cache from an earlier call to the function under test, the
    function under test will return identical output and will not delegate to
    an expensive transformer function."""
    embeddings_cache = embeddings_cache.withColumnsRenamed({"key": "_text_to_embed", "value": "embedding"})
    pre_embedding.cache()
    kwargs = {
        "df": pre_embedding,
        "transformer": return_constant,  # the function will get called as many times as there are partitions
        "max_input_len": 10,
        "input_features": ("name", "category"),
        "scope": scope,
        "model": model,
        "new_colname": "embedding",
    }
    embedded, cache_v2 = nodes.create_node_embeddings(
        cache=embeddings_cache,
        **kwargs,
    )

    re_embedded, cache_v3 = nodes.create_node_embeddings(
        cache=cache_v2.cache(),
        **kwargs,
    )

    assertDataFrameEqual(embedded, re_embedded)
    assertDataFrameEqual(cache_v3, cache_v2)


def test_reduced_embedding_calls_in_presence_of_a_cache(
    pre_embedding: ps.DataFrame, embeddings_cache: ps.DataFrame, mock_encoder: Mock, mock_encoder2: Mock
):
    embedded, cache_v2 = nodes.create_node_embeddings(
        df=pre_embedding.cache(),
        cache=embeddings_cache,
        batch_size=1,
        transformer=mock_encoder,
        max_input_len=10,
        input_features=("name", "category"),
    )

    assert mock_encoder.embed.called_once()

    pre_embedding.union(pre_embedding.sparkSession.createDataFrame([(3, "c", "C")], schema=pre_embedding.schema))
    # New run, but with an extended cache. Pass in a new mock, as we want to verify it has not been called.
    re_embedded, cache_v3 = nodes.create_node_embeddings(
        df=pre_embedding,
        cache=cache_v2.cache(),
        batch_size=1,
        transformer=mock_encoder2,
        max_input_len=10,
        input_features=("name", "category"),
    )

    mock_encoder2.embed.assert_called_once_with(["cC"])

    assertDataFrameEqual(embedded, re_embedded)
    assertDataFrameEqual(cache_v3, cache_v2)

    pass


def test_embedding_cache_gets_bootstrapped(
    pre_embedding: ps.DataFrame,
    embeddings_cache: ps.DataFrame,
    mock_encoder: Mock,
    mock_encoder2: Mock,
    scope: str,
    model: str,
):
    empty_cache = embeddings_cache.limit(0)
    # this is not bootstrapping, since Kedro will barf if the dataset doesn't exist
    embedded, cache_v2 = nodes.create_node_embeddings(
        df=pre_embedding.cache(),
        cache=empty_cache,  # for bootstrapping: create (maybe it exists already?) a LazySparkDataset and on entry in create_node_embeddings wrap it with a try except
        batch_size=1,
        transformer=mock_encoder,
        max_input_len=10,
        input_features=("name", "category"),
    )

    assert mock_encoder.embed.called_once_with(["aA", "bB"])
    expected_cache = pre_embedding.sparkSession.createDataFrame(
        [
            ("foo", "bar", "aA", [1]),
            ("foo", "bar", "bB", [1]),
        ]
    )
    assertDataFrameEqual(cache_v2, expected_cache)


def test_embeddings_batch_size_determines_number_of_network_calls():
    # 5 rows, batch_size=1-> 5 calls. 5 rows, batch_size=2, -> 5calls or less (depends on number of partitions)
    pass
