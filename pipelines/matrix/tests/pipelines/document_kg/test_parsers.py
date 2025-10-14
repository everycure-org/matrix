import pandas as pd
import pytest
from matrix.pipelines.document_kg.parsers import (
    InforesParser,
    ReusableDataParser,
)


@pytest.fixture
def infores_parser():
    """Create InforesParser instance."""
    return InforesParser(
        name="infores",
        id_column="id",
        extracted_metadata=["id", "name", "description", "status"],
        data_path="information_resources",
    )


@pytest.fixture
def infores_sample_data():
    """Sample infores data."""
    return {
        "information_resources": [
            {
                "id": "infores:test_source",
                "name": "Test Source",
                "description": "A test information resource",
                "status": "released",
                "extra_field": "should be ignored",
            },
            {
                "id": "infores:another_source",
                "name": "Another Source",
                "description": "Another test resource",
                "status": "deprecated",
            },
            {
                "name": "No ID Source",
                "description": "This should be skipped",
            },
        ]
    }


@pytest.fixture
def reusabledata_parser():
    """Create ReusableDataParser instance."""
    return ReusableDataParser(
        name="reusabledata",
        id_column="updated_id",
        original_id_column="id",
        extracted_metadata=["id", "description", "license"],
    )


@pytest.fixture
def reusabledata_sample_data():
    """Sample reusabledata JSON."""
    return [
        {
            "id": "rd:source1",
            "description": "First reusable data source",
            "license": "CC-BY-4.0",
        },
        {
            "id": "rd:source2",
            "description": "Second reusable data source",
            "license": "MIT",
        },
    ]


@pytest.fixture
def mapping_df():
    """Sample SSSOM mapping."""
    return pd.DataFrame(
        {
            "subject_id": ["rd:source1", "rd:source2"],
            "object_id": ["infores:mapped1", "infores:mapped2"],
        }
    )


def test_infores_parse_basic(infores_parser, infores_sample_data):
    """Test parsing basic infores data."""
    result = infores_parser.parse(infores_sample_data)

    assert len(result) == 2
    assert "test_source" in result
    assert "infores" in result["test_source"]
    assert result["test_source"]["infores"]["id"] == "infores:test_source"
    assert result["test_source"]["infores"]["name"] == "Test Source"


def test_infores_strips_prefix_and_filters_fields(infores_parser, infores_sample_data):
    """Test that infores: prefix is stripped and extra fields are filtered."""
    result = infores_parser.parse(infores_sample_data)

    assert "test_source" in result
    assert "infores:test_source" not in result
    assert result["test_source"]["infores"]["id"] == "infores:test_source"
    assert "extra_field" not in result["test_source"]["infores"]


def test_infores_skips_empty_ids(infores_parser, infores_sample_data):
    """Test that records without IDs are skipped."""
    result = infores_parser.parse(infores_sample_data)

    assert len(result) == 2
    assert all("id" in result[key]["infores"] for key in result)


def test_reusabledata_parse_with_mapping(reusabledata_parser, reusabledata_sample_data, mapping_df):
    """Test parsing with ID mapping."""
    result = reusabledata_parser.parse(reusabledata_sample_data, mapping_data=mapping_df)

    assert "mapped1" in result
    assert "mapped2" in result
    assert result["mapped1"]["reusabledata"]["updated_id"] == "infores:mapped1"


def test_reusabledata_parse_partial_mapping(reusabledata_parser, reusabledata_sample_data):
    """Test that unmapped IDs are kept as-is."""
    partial_mapping = pd.DataFrame({"subject_id": ["rd:source1"], "object_id": ["infores:mapped1"]})

    result = reusabledata_parser.parse(reusabledata_sample_data, mapping_data=partial_mapping)

    assert "mapped1" in result
    assert "source2" in result or "rd:source2" in result
