import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType

from matrix.pipelines.modelling.nodes import prefilter_nodes


@pytest.fixture
def spark():
    return SparkSession.builder.getOrCreate()


@pytest.fixture
def base_test_data(spark):
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


@pytest.fixture
def ground_truth_data(spark):
    gt_data = [("node1", "node2", 1), ("node3", "node4", 0)]
    gt_schema = StructType(
        [
            StructField("source", StringType(), False),
            StructField("target", StringType(), False),
            StructField("y", IntegerType(), False),
        ]
    )
    return spark.createDataFrame(gt_data, schema=gt_schema)


def test_prefilter_excludes_non_drug_disease_nodes(spark, base_test_data, ground_truth_data):
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


def test_prefilter_correctly_identifies_drug_nodes(spark, base_test_data, ground_truth_data):
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


def test_prefilter_correctly_identifies_disease_nodes(spark, base_test_data, ground_truth_data):
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


def test_prefilter_correctly_identifies_ground_truth_positives(spark, base_test_data, ground_truth_data):
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
