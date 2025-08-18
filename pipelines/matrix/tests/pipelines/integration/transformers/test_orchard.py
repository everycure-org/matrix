import pytest
from matrix.pipelines.integration.transformers.orchard import OrchardTransformer
from pyspark.sql.types import StringType, StructField, StructType


@pytest.fixture
def edges_df(spark):
    """Create sample edges dataframe."""
    schema = StructType(
        [
            StructField("drug_kg_node_id", StringType(), True),
            StructField("disease_kg_node_id", StringType(), True),
            StructField("last_update", StringType(), True),
            StructField("latest_status", StringType(), True),
            StructField("latest_source", StringType(), True),
            StructField("latest_directionality", StringType(), True),
            StructField("latest_reason", StringType(), True),
        ]
    )

    data = [
        # Semmed edges
        ("curie1", "curie2", "2022-11-31 00:00:00", "SAB_ENDORSED", "MATRIX", "POSITIVE", "NULL"),
        ("curie3", "curie4", "2022-12-31 00:00:00", "SAB_ENDORSED", "CROWDSOURCED", "POSITIVE", "NULL"),
        ("curie3", "curie2", "2022-09-31 00:00:00", "MEDICAL_REVIEW", "MATRIX", "POSITIVE", "NULL"),
        ("curie3", "curie5", "2022-10-31 00:00:00", "MEDICAL_REVIEW", "CROWDSOURCED", "POSITIVE", "NULL"),
        ("curie6", "curie7", "2022-12-31 00:00:00", "ARCHIVED", "MATRIX", "NEGATIVE", "POOR_STRATEGIC_RATIONALE"),
        ("curie8", "curie7", "2022-12-31 00:00:00", "ARCHIVED", "MATRIX", "NEGATIVE", "POOR_BIOMEDICAL_RATIONALE"),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def pair_flags():
    """Create sample pair flags."""
    return {
        "high_evidence_matrix": {
            "latest_source": ["MATRIX"],
            "latest_status": ["DEEP_DIVE", "SAB_ENDORSED", "HOLDING"],
        },
        "high_evidence_crowdsourced": {
            "latest_source": ["CROWDSOURCED"],
            "latest_status": ["DEEP_DIVE", "SAB_ENDORSED", "HOLDING"],
        },
        "mid_evidence_matrix": {
            "latest_source": ["MATRIX"],
            "latest_status": ["DEEP_DIVE", "SAB_ENDORSED", "HOLDING", "MEDICAL_REVIEW"],
        },
        "mid_evidence_crowdsourced": {
            "latest_source": ["CROWDSOURCED"],
            "latest_status": ["DEEP_DIVE", "SAB_ENDORSED", "HOLDING", "MEDICAL_REVIEW"],
        },
        "archive_biomedical_review": {"latest_source": ["MATRIX"], "latest_reason": ["POOR_BIOMEDICAL_RATIONALE"]},
    }


@pytest.fixture
def pair_flags_error():
    """Create sample pair flags."""
    return {
        "high_evidence_matrix": {
            "latest_typo_source": ["MATRIX"],
            "latest_status": ["DEEP_DIVE", "SAB_ENDORSED", "HOLDING"],
        },
    }


def test_orchard_flags(edges_df, pair_flags, pair_flags_error):
    """Test that the flags are correctly applied to the edges dataframe."""
    transformer = OrchardTransformer(version="v0.0.0", pair_flags=pair_flags)
    output = (transformer.transform(edges_df))["edges"]
    expected_columns = [
        "high_evidence_matrix",
        "high_evidence_crowdsourced",
        "mid_evidence_matrix",
        "mid_evidence_crowdsourced",
        "archive_biomedical_review",
    ]
    assert all(column in output.columns for column in expected_columns)
    assert len(list(output.columns)) == 11
    assert output.count() == edges_df.count()

    # Check if logic is applied correctly
    assert output.filter(output.high_evidence_matrix == 1).count() == 1
    assert output.filter(output.high_evidence_crowdsourced == 1).count() == 1
    assert output.filter(output.mid_evidence_matrix == 1).count() == 2
    assert output.filter(output.mid_evidence_crowdsourced == 1).count() == 2
    assert output.filter(output.archive_biomedical_review == 1).count() == 1

    # First and last row checks
    assert output.select(expected_columns).first() == (1, 0, 1, 0, 0)

    # Check if errors out correctly when column not found
    with pytest.raises(Exception) as e_info:
        transformer = OrchardTransformer(version="v0.0.0", pair_flags=pair_flags_error)
        transformer.transform(edges_df)
