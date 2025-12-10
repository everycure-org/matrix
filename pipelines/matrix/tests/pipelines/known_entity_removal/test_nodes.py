import pyspark.sql as ps
import pytest
from matrix.pipelines.known_entity_removal.mondo_ontology import OntologyTest
from matrix.pipelines.known_entity_removal.nodes import (
    apply_mondo_expansion,
    concatenate_datasets,
    create_known_entity_matrix,
)
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def sample_dataset_1(spark: ps.SparkSession) -> ps.DataFrame:
    """Create a sample dataset with positive and negative pairs."""
    data = [
        ("drug_a", "disease_a", 1),
        ("drug_a", "disease_b", 0),
        ("drug_b", "disease_a", 1),
    ]
    return spark.createDataFrame(data, schema=["subject", "object", "y"])


@pytest.fixture
def sample_dataset_2(spark: ps.SparkSession) -> ps.DataFrame:
    """Create another sample dataset with positive and negative pairs."""
    data = [
        ("drug_b", "disease_b", 1),
        ("drug_c", "disease_a", 0),
        ("drug_a", "disease_a", 1),  # Duplicate with dataset_1
    ]
    return spark.createDataFrame(data, schema=["subject", "object", "y"])


@pytest.fixture
def sample_drug_list(spark: ps.SparkSession) -> ps.DataFrame:
    """Create a sample drug list."""
    data = [
        ("drug_a", "ec_id_a"),
        ("drug_b", "ec_id_b"),
    ]
    return spark.createDataFrame(data, schema=["id", "ec_id"])


@pytest.fixture
def sample_disease_list(spark: ps.SparkSession) -> ps.DataFrame:
    """Create a sample disease list."""
    data = [
        ("disease_a",),
        ("disease_b",),
    ]
    return spark.createDataFrame(data, schema=["core_id"])


def test_concatenate_datasets(spark: ps.SparkSession, sample_dataset_1, sample_dataset_2):
    # Given two datasets with mixed inclusion parameters
    datasets_to_include = {
        "dataset_1": {"positives": True, "negatives": True},
        "dataset_2": {"positives": True, "negatives": False},
    }
    all_datasets = {
        "dataset_1": sample_dataset_1,
        "dataset_2": sample_dataset_2,
    }

    # When we concatenate the datasets
    result = concatenate_datasets(datasets_to_include=datasets_to_include, **all_datasets)

    # Then we get unique pairs with renamed columns, filtering by inclusion parameters
    expected = spark.createDataFrame(
        [
            ("drug_a", "disease_a"),
            ("drug_a", "disease_b"),
            ("drug_b", "disease_a"),
            ("drug_b", "disease_b"),
        ],
        schema=["drug_id", "disease_id"],
    )
    assertDataFrameEqual(result, expected)


def test_apply_mondo_expansion(spark: ps.SparkSession):
    # Given a concatenated ground truth with diseases and the OntologyTest class
    concatenated_ground_truth = spark.createDataFrame(
        [
            ("drug_a", "disease_a"),
            ("drug_b", "disease_b"),
        ],
        schema=["drug_id", "disease_id"],
    )
    mondo_ontology = OntologyTest()

    # When we apply Mondo expansion
    result = apply_mondo_expansion(mondo_ontology=mondo_ontology, concatenated_ground_truth=concatenated_ground_truth)

    # Then we get expanded pairs with equivalent disease IDs (ancestor and descendant)
    expected = spark.createDataFrame(
        [
            ("drug_a", "disease_a"),
            ("drug_a", "disease_a_ancestor"),
            ("drug_a", "disease_a_descendant"),
            ("drug_b", "disease_b"),
            ("drug_b", "disease_b_ancestor"),
            ("drug_b", "disease_b_descendant"),
        ],
        schema=["drug_id", "disease_id"],
    )
    assertDataFrameEqual(result, expected)


def test_create_known_entity_matrix(spark: ps.SparkSession, sample_drug_list, sample_disease_list):
    # Given drug list, disease list, and expanded ground truth with known entities
    expanded_ground_truth = spark.createDataFrame(
        [("drug_a", "disease_a")],
        schema=["drug_id", "disease_id"],
    )

    # When we create the known entity matrix
    result = create_known_entity_matrix(
        drug_list=sample_drug_list,
        disease_list=sample_disease_list,
        expanded_ground_truth=expanded_ground_truth,
    )

    # Then we get a cross join with known entities marked as True and others as False
    expected = spark.createDataFrame(
        [
            ("drug_a", "disease_a", "ec_id_a", True),
            ("drug_a", "disease_b", "ec_id_a", False),
            ("drug_b", "disease_a", "ec_id_b", False),
            ("drug_b", "disease_b", "ec_id_b", False),
        ],
        schema=["drug_translator_id", "target", "ec_drug_id", "is_known_entity"],
    )
    assertDataFrameEqual(result, expected)
