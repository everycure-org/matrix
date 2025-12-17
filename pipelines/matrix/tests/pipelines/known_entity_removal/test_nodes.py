import pandas as pd
import pyspark.sql as ps
import pytest
from matrix.pipelines.known_entity_removal.mondo_ontology import OntologyTest
from matrix.pipelines.known_entity_removal.nodes import (
    apply_mondo_expansion,
    concatenate_datasets,
    create_known_entity_matrix,
    preprocess_orchard_pairs,
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


@pytest.fixture
def sample_orchard_pairs(spark: ps.SparkSession) -> dict[str, pd.DataFrame]:
    """Create a sample orchard pairs dataframe."""
    orchard_pairs = pd.DataFrame(
        {
            "drug_name": ["DrugA", "DrugB", "DrugA"],
            "disease_name": ["DiseaseA", "DiseaseB", "DiseaseA"],
            "report_date": ["2021-03-01", "2021-03-01", "2021-04-01"],
            "last_created_at_at_report_date": [
                "2020-06-20",
                "2020-06-24",
                "2020-09-21",
            ],
            "drug_kg_node_id": ["EC:00001", "EC:00002", "EC:00003"],
            "disease_kg_node_id": ["RTX:00001", "RTX:00002", "RTX:00003"],
            "status_transitions_up_to_report_date": [
                "UNKNOWN > TRIAGE > SAC_ENDORSED",
                "UNKNOWN > TRIAGE > DEEP_DIVE > ARCHIVED",
                "UNKNOWN > TRIAGE> MEDICAL_REVIEW > DEEP_DIVE > ARCHIVED",
            ],
            "depriortization_reason_at_report_date": [
                None,
                "DRUG_ON_LABEL_FOR_DISEASE",
                "DRUG_WIDELY_USED_OFF_LABEL",
            ],
        }
    )
    orchard_report_date = "2021-04-01"
    return {"orchard_pairs": orchard_pairs, "orchard_report_date": orchard_report_date}


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


def test_preprocess_orchard_pairs(sample_orchard_pairs):
    # Given a mock Orchard pairs by month dataframe and a fixed report date
    orchard_pairs = sample_orchard_pairs["orchard_pairs"]
    orchard_report_date = sample_orchard_pairs["orchard_report_date"]

    # When we preprocess the Orchard pairs
    result = preprocess_orchard_pairs(orchard_pairs=orchard_pairs, orchard_report_date=orchard_report_date)
    result_with_latest = preprocess_orchard_pairs(orchard_pairs=orchard_pairs, orchard_report_date="latest")

    # Then the report date info should be the same
    report_date = result["report_date_info"].iloc[0]["orchard_data_report_date"]
    report_date_latest = result_with_latest["report_date_info"].iloc[0]["orchard_data_report_date"]
    assert report_date == pd.Timestamp("2021-04-01")
    assert report_date_latest == pd.Timestamp("2021-04-01")

    # Only one row for 2021-04-01
    processed_pairs = result["processed_orchard_pairs"]
    assert len(processed_pairs) == 1

    # And the labels should have correct boolean values
    assert processed_pairs.iloc[0]["reached_triage"] == True
    assert processed_pairs.iloc[0]["reached_med_review"] == True
    assert processed_pairs.iloc[0]["reached_deep_dive"] == True
    assert processed_pairs.iloc[0]["reached_sac"] == False
    assert processed_pairs.iloc[0]["archived_known_off_label"] == True
    assert processed_pairs.iloc[0]["archived_known_on_label"] == False
    assert processed_pairs.iloc[0]["archived_known_entity"] == True
    assert processed_pairs.iloc[0]["triaged_not_known_entity"] == False  # Archived as known entity
