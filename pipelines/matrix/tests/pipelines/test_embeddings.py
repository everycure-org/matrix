from pathlib import Path
from random import choice
from typing import Callable, Iterable, Iterator, Sequence, Tuple, Type, TypeVar
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pyspark.sql as ps
import pyspark.sql.types as ty
import pytest
from matrix.pipelines.embeddings import nodes
from matrix.pipelines.embeddings.encoders import AttributeEncoder
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
def raw_embeddable_nodes(spark: ps.SparkSession) -> ps.DataFrame:
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
def embeddable_nodes(spark: ps.SparkSession, text_col: str) -> ps.DataFrame:
    """Provides the data with the texts in a format that will be sent unaltered to the embedding function."""
    return spark.createDataFrame(
        [
            (1, "aA"),
            (2, "aB"),
            (3, "aC"),
            (4, "aC"),  # intentional duplicate text
            (5, "aD"),
        ],
        schema=("id", text_col),
    )


@pytest.fixture
def scope():
    return "foo"


@pytest.fixture
def model():
    return "bar"


@pytest.fixture
def text_col():
    return "text"


@pytest.fixture
def embedding_col():
    return "embedding"


@pytest.fixture
def embeddings_cache(scope: str, model: str, text_col: str, embedding_col: str, spark: ps.SparkSession) -> ps.DataFrame:
    return spark.createDataFrame(
        [
            (scope, model, "aA", [1.0]),
            (scope, model, "cC", [1.0]),
        ],
        schema=("scope", "model", text_col, embedding_col),
    )


@pytest.fixture
def scope_ltd_embeddings_cache(text_col: str, embedding_col: str, spark: ps.SparkSession) -> ps.DataFrame:
    return spark.createDataFrame([("aA", [1.0]), ("cC", [1.0])], schema=(text_col, embedding_col))


@pytest.fixture
def return_constant(tmp_path: Path):
    # The function below will be used in Spark's executors. To do so, the
    # function gets serialized, which means mocks get recreated and one can no
    # longer use their `assert_called_once` and similar methods. The following
    # allows us to track how many times the function actually got called, when
    # dealing with subprocesses that unpickle this function.
    def embedder(docs: Iterable[str]) -> Iterator[nodes.ResolvedEmbedding]:
        for doc in docs:
            (tmp_path / uuid4().hex).touch(exist_ok=False)
            yield doc, [1.0] * choice((1, 2))

    yield embedder, tmp_path


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


def test_re_embedding_is_a_noop(
    raw_embeddable_nodes: ps.DataFrame,
    embeddings_cache: ps.DataFrame,
    return_constant: tuple[Callable, Path],
    scope: str,
    model: str,
    embedding_col: str,
    text_col: str,
):
    """Given a cache from an earlier call to the function under test, the
    function under test will return identical output and will not delegate to
    an expensive transformer function."""
    m = Mock(spec=AttributeEncoder)
    m.apply = return_constant[0]

    raw_embeddable_nodes.cache()
    kwargs = {
        "df": raw_embeddable_nodes,
        "transformer": m,  # the function will get called as many times as there are partitions
        "max_input_len": 10,
        "input_features": ("name", "category"),
        "scope": scope,
        "model": model,
        "new_colname": embedding_col,
        "embeddings_pkey": text_col,
    }
    embedded, cache_v2 = nodes.create_node_embeddings(
        cache=embeddings_cache,
        **kwargs,
    )

    # Force materialization and thus calling of the transformer/embedder,
    # while also verifying that the cache grew in size.
    assert cache_v2.count() > embeddings_cache.count()
    assert next(return_constant[1].glob("*"))  # The embedder function got called at least once

    for f in return_constant[1].glob("*"):
        f.unlink()  # The fixture only creates files in there. Empty the dir for a new run.

    re_embedded, cache_v3 = nodes.create_node_embeddings(
        cache=cache_v2.cache(),
        **kwargs,
    )
    assert list(return_constant[1].glob("*")) == []  # This indicates we did not step into the external embedder!

    assertDataFrameEqual(embedded, re_embedded)
    assertDataFrameEqual(cache_v3, cache_v2)


def test_reduced_embedding_calls_in_presence_of_a_cache(
    embeddable_nodes: ps.DataFrame,
    scope_ltd_embeddings_cache: ps.DataFrame,
    return_constant: tuple[Callable, Path],
    text_col: str,
    embedding_col: str,
):
    embedded, cache_v2 = nodes.lookup_embeddings(
        df=embeddable_nodes.cache(),
        cache=scope_ltd_embeddings_cache,
        embedder=return_constant[0],
        text_colname=text_col,
        new_colname=embedding_col,
    )

    embedded.count()  # Force execution of the query plan.
    unique_texts, cache_keys = (
        set(_[0] for _ in df.select(text_col).collect()) for df in (embeddable_nodes, scope_ltd_embeddings_cache)
    )
    # For each unique_text that is not in the cache, a lookup will be made.
    assert len(list(return_constant[1].glob("*"))) == len(unique_texts.difference(cache_keys)) < len(unique_texts)


def test_embedding_cache_gets_bootstrapped(
    raw_embeddable_nodes: ps.DataFrame,
    embeddings_cache: ps.DataFrame,
    mock_encoder: Mock,
    mock_encoder2: Mock,
    scope: str,
    model: str,
):
    empty_cache = embeddings_cache.limit(0)
    # this is not bootstrapping, since Kedro will barf if the dataset doesn't exist
    embedded, cache_v2 = nodes.create_node_embeddings(
        df=raw_embeddable_nodes.cache(),
        cache=empty_cache,  # for bootstrapping: create (maybe it exists already?) a LazySparkDataset and on entry in create_node_embeddings wrap it with a try except
        transformer=mock_encoder,
        max_input_len=10,
        input_features=("name", "category"),
    )

    assert mock_encoder.embed.called_once_with(["aA", "bB"])
    expected_cache = raw_embeddable_nodes.sparkSession.createDataFrame(
        [
            ("foo", "bar", "aA", [1]),
            ("foo", "bar", "bB", [1]),
        ]
    )
    assertDataFrameEqual(cache_v2, expected_cache)


def test_embeddings_batch_size_determines_number_of_network_calls():
    # 5 rows, batch_size=1-> 5 calls. 5 rows, batch_size=2, -> 5calls or less (depends on number of partitions)
    # This would apply to the LangChainEncoder
    pass
