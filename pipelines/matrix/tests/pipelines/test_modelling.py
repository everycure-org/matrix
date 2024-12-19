import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType

from matrix.pipelines.modelling.nodes import prefilter_nodes


@pytest.fixture
def spark():
    return SparkSession.builder.getOrCreate()


def test_prefilter_nodes(spark):
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
    nodes_df = spark.createDataFrame(nodes_data, schema=nodes_schema)

    # Ground truth data
    gt_data = [("node1", "node2", 1), ("node3", "node4", 0)]
    gt_schema = StructType(
        [
            StructField("source", StringType(), False),
            StructField("target", StringType(), False),
            StructField("y", IntegerType(), False),
        ]
    )
    gt_df = spark.createDataFrame(gt_data, schema=gt_schema)

    # Define drug and disease types
    drug_types = ["drug_type1", "drug_type2"]
    disease_types = ["disease_type1"]

    # Call the function
    result = prefilter_nodes(
        full_nodes=nodes_df,  # Not used in current implementation
        nodes=nodes_df,
        gt=gt_df,
        drug_types=drug_types,
        disease_types=disease_types,
    )

    # Convert result to list for easier assertions
    result_list = result.collect()

    # Assertions
    assert len(result_list) == 3  # Should only include drug/disease nodes

    # Check that node1 (drug) is present
    node1 = next(row for row in result_list if row.id == "node1")
    assert node1.is_drug is True
    assert node1.is_disease is False
    assert node1.is_ground_pos is True

    # Check that node2 (disease) is present
    node2 = next(row for row in result_list if row.id == "node2")
    assert node2.is_drug is False
    assert node2.is_disease is True
    assert node2.is_ground_pos is True

    # Check that node4 (drug) is present
    node4 = next(row for row in result_list if row.id == "node4")
    assert node4.is_drug is True
    assert node4.is_disease is False
    assert node4.is_ground_pos is False

    # Check that node3 (neither drug nor disease) is not present
    assert not any(row.id == "node3" for row in result_list)
