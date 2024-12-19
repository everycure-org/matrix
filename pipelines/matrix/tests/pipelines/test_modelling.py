import numpy as np
import pytest
import pandera
import pandas as pd

from sklearn.model_selection import KFold

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, FloatType
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

from matrix.pipelines.modelling.nodes import prefilter_nodes
from matrix.pipelines.modelling.nodes import make_splits
from matrix.pipelines.modelling.nodes import attach_embeddings


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


@pytest.fixture(scope="module")
def base_test_data(spark: SparkSession) -> DataFrame:
    # Create test data
    nodes_data = [
        ("node1", ["drug_type1"], None),
        ("node2", ["disease_type1"], None),
        ("node3", ["other_type"], None),
        ("node4", ["drug_type2", "other_type"], None),
    ]
    nodes_schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("all_categories", ArrayType(StringType()), True),
            StructField("topological_embedding", ArrayType(StringType()), True),
        ]
    )
    return spark.createDataFrame(nodes_data, schema=nodes_schema)


@pytest.fixture(scope="module")
def ground_truth_data(spark: SparkSession) -> DataFrame:
    gt_data = [("node1", "node2", 1), ("node3", "node4", 0)]
    gt_schema = StructType(
        [
            StructField("source", StringType(), False),
            StructField("target", StringType(), False),
            StructField("y", IntegerType(), False),
        ]
    )
    return spark.createDataFrame(gt_data, schema=gt_schema)


def test_prefilter_excludes_non_drug_disease_nodes(base_test_data: DataFrame, ground_truth_data: DataFrame) -> None:
    result = prefilter_nodes(
        full_nodes=base_test_data,
        nodes=base_test_data,
        gt=ground_truth_data,
        drug_types=["drug_type1", "drug_type2"],
        disease_types=["disease_type1"],
    )

    result_list = result.collect()
    assert len(result_list) == 3  # Should only include drug/disease nodes
    assert not any(row.id == "node3" for row in result_list)


def test_prefilter_correctly_identifies_drug_nodes(base_test_data: DataFrame, ground_truth_data: DataFrame) -> None:
    result = prefilter_nodes(
        full_nodes=base_test_data,
        nodes=base_test_data,
        gt=ground_truth_data,
        drug_types=["drug_type1", "drug_type2"],
        disease_types=["disease_type1"],
    )

    result_list = result.collect()

    # Check drug nodes
    node1 = next(row for row in result_list if row.id == "node1")
    assert node1.is_drug is True
    assert node1.is_disease is False

    node4 = next(row for row in result_list if row.id == "node4")
    assert node4.is_drug is True
    assert node4.is_disease is False


def test_prefilter_correctly_identifies_disease_nodes(base_test_data: DataFrame, ground_truth_data: DataFrame) -> None:
    result = prefilter_nodes(
        full_nodes=base_test_data,
        nodes=base_test_data,
        gt=ground_truth_data,
        drug_types=["drug_type1", "drug_type2"],
        disease_types=["disease_type1"],
    )

    result_list = result.collect()

    node2 = next(row for row in result_list if row.id == "node2")
    assert node2.is_drug is False
    assert node2.is_disease is True


def test_prefilter_correctly_identifies_ground_truth_positives(
    base_test_data: DataFrame, ground_truth_data: DataFrame
) -> None:
    result = prefilter_nodes(
        full_nodes=base_test_data,
        nodes=base_test_data,
        gt=ground_truth_data,
        drug_types=["drug_type1", "drug_type2"],
        disease_types=["disease_type1"],
    )

    result_list = result.collect()

    # Check ground truth positive nodes
    node1 = next(row for row in result_list if row.id == "node1")
    assert node1.is_ground_pos is True

    node2 = next(row for row in result_list if row.id == "node2")
    assert node2.is_ground_pos is True

    node4 = next(row for row in result_list if row.id == "node4")
    assert node4.is_ground_pos is False


@pytest.fixture(scope="module")
def sample_pairs_df(spark: SparkSession) -> DataFrame:
    pairs_data = [("node1", "node2", 1), ("node3", "node4", 0), ("node5", "node6", 1)]
    pairs_schema = StructType(
        [
            StructField("source", StringType(), False),
            StructField("target", StringType(), False),
            StructField("y", IntegerType(), False),
        ]
    )
    return spark.createDataFrame(pairs_data, schema=pairs_schema)


@pytest.fixture(scope="module")
def sample_nodes_df(spark: SparkSession) -> DataFrame:
    nodes_data = [
        ("node1", [0.1, 0.2, 0.3]),
        ("node2", [0.4, 0.5, 0.6]),
        ("node3", [0.7, 0.8, 0.9]),
        ("node4", [1.0, 1.1, 1.2]),
        ("node5", [1.3, 1.4, 1.5]),
        ("node6", [1.6, 1.7, 1.8]),
    ]
    nodes_schema = StructType(
        [StructField("id", StringType(), False), StructField("topological_embedding", ArrayType(FloatType()), False)]
    )
    return spark.createDataFrame(nodes_data, schema=nodes_schema)


def test_attach_embeddings_successful(sample_pairs_df: DataFrame, sample_nodes_df: DataFrame) -> None:
    """Test successful attachment of embeddings to pairs."""
    result = attach_embeddings(sample_pairs_df, sample_nodes_df)

    # Check schema
    assert "source_embedding" in result.columns
    assert "target_embedding" in result.columns
    assert "y" in result.columns

    # Check number of rows preserved
    assert result.count() == sample_pairs_df.count()

    first_row = result.filter(F.col("source") == "node1").first()

    assert np.allclose(first_row["source_embedding"], [0.1, 0.2, 0.3])
    assert np.allclose(first_row["target_embedding"], [0.4, 0.5, 0.6])


def test_attach_embeddings_missing_nodes(
    spark: SparkSession, sample_pairs_df: DataFrame, sample_nodes_df: DataFrame
) -> None:
    """Test handling of pairs with missing nodes."""
    # Add a pair with non-existent nodes
    new_pair = spark.createDataFrame([("nodeX", "nodeY", 1)], schema=sample_pairs_df.schema)
    pairs_with_missing = sample_pairs_df.union(new_pair)

    with pytest.raises(pandera.errors.SchemaError):
        attach_embeddings(pairs_with_missing, sample_nodes_df)


def test_attach_embeddings_empty_pairs(
    spark: SparkSession, sample_pairs_df: DataFrame, sample_nodes_df: DataFrame
) -> None:
    """Test handling of empty pairs dataframe."""
    empty_pairs = spark.createDataFrame([], schema=sample_pairs_df.schema)
    result = attach_embeddings(empty_pairs, sample_nodes_df)
    assert result.count() == 0


def test_attach_embeddings_schema_validation(spark: SparkSession, sample_pairs_df: DataFrame) -> None:
    """Test schema validation with invalid node embeddings."""
    # Create nodes with invalid embedding type (integers instead of floats)
    invalid_nodes_data = [("node1", [1, 2, 3]), ("node2", [4, 5, 6])]
    invalid_nodes_schema = StructType(
        [StructField("id", StringType(), False), StructField("topological_embedding", ArrayType(IntegerType()), False)]
    )
    invalid_nodes_df = spark.createDataFrame(invalid_nodes_data, schema=invalid_nodes_schema)

    with pytest.raises(pandera.errors.SchemaError):
        attach_embeddings(sample_pairs_df, invalid_nodes_df)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "source": ["A", "B", "C", "D", "E", "F"],
            "source_embedding": [np.array([1, 2])] * 6,
            "target": ["X", "Y", "Z", "W", "V", "U"],
            "target_embedding": [np.array([3, 4])] * 6,
            "y": [1, 0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def simple_splitter():
    """Create a simple K-fold splitter."""
    return KFold(n_splits=2, shuffle=True, random_state=42)


def test_make_splits_basic_functionality(sample_data, simple_splitter):
    """Test basic functionality of make_splits."""
    result = make_splits(data=sample_data, splitter=simple_splitter)

    # Check that all required columns are present
    required_columns = ["source", "source_embedding", "target", "target_embedding", "iteration", "split"]
    assert all(col in result.columns for col in required_columns)

    # Check that we have the same number of rows as input
    assert len(result) == len(sample_data) * 2  # 2 iterations

    # Check that splits are properly labeled
    assert set(result["split"].unique()) == {"TRAIN", "TEST"}

    # Check iterations
    assert set(result["iteration"].unique()) == {0, 1}


def test_make_splits_data_integrity(sample_data, simple_splitter):
    """Test that data values are preserved after splitting."""
    result = make_splits(data=sample_data, splitter=simple_splitter)

    # Check that original values are preserved
    assert set(result["source"].unique()) == set(sample_data["source"].unique())
    assert set(result["target"].unique()) == set(sample_data["target"].unique())


def test_make_splits_schema_validation(sample_data, simple_splitter):
    """Test schema validation with invalid data."""
    # Create invalid data missing required columns
    invalid_data = sample_data.drop(columns=["source_embedding"])

    with pytest.raises(pandera.errors.SchemaError):
        make_splits(data=invalid_data, splitter=simple_splitter)


def test_make_splits_train_test_distribution(sample_data, simple_splitter):
    """Test that each fold has both train and test data."""
    result = make_splits(data=sample_data, splitter=simple_splitter)

    for iteration in result["iteration"].unique():
        iteration_data = result[result["iteration"] == iteration]

        # Check that both train and test splits exist
        assert "TRAIN" in iteration_data["split"].values
        assert "TEST" in iteration_data["split"].values

        # Check that splits are mutually exclusive
        train_indices = set(iteration_data[iteration_data["split"] == "TRAIN"].index)
        test_indices = set(iteration_data[iteration_data["split"] == "TEST"].index)
        assert len(train_indices.intersection(test_indices)) == 0


def test_make_splits_empty_data(simple_splitter):
    """Test behavior with empty input data."""
    empty_data = pd.DataFrame(columns=["source", "source_embedding", "target", "target_embedding", "y"])

    with pytest.raises(ValueError):
        make_splits(data=empty_data, splitter=simple_splitter)
