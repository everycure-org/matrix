import os
from unittest.mock import patch

import pytest
from matrix.resolvers import get_kg_raw_path_for_source
from matrix.utils.hook_utilities import disable_private_datasets, generate_dynamic_pipeline_mapping


@pytest.fixture
def integration_mapping():
    return {
        "integration": [
            {"name": "public_source_1", "integrate_in_kg": True, "is_private": False},
            {"name": "private_source", "integrate_in_kg": True, "is_private": True},
            {"name": "public_source_2", "integrate_in_kg": True, "is_private": False},
        ]
    }


@pytest.mark.parametrize(
    "include_private_datasets,expected_sources",
    [
        (
            "",
            [
                {"name": "public_source_1", "integrate_in_kg": True, "is_private": False},
                {"name": "public_source_2", "integrate_in_kg": True, "is_private": False},
            ],
        ),
        (
            "1",
            [
                {"name": "public_source_1", "integrate_in_kg": True, "is_private": False},
                {"name": "private_source", "integrate_in_kg": True, "is_private": True},
                {"name": "public_source_2", "integrate_in_kg": True, "is_private": False},
            ],
        ),
    ],
    ids=["dev_environment", "prod_environment"],
)
def test_integration_sources_filtering(integration_mapping, monkeypatch, include_private_datasets, expected_sources):
    monkeypatch.setenv("INCLUDE_PRIVATE_DATASETS", include_private_datasets)
    result = disable_private_datasets(integration_mapping)
    result_nested = disable_private_datasets(generate_dynamic_pipeline_mapping(integration_mapping))
    assert result["integration"] == expected_sources
    assert result_nested["integration"] == expected_sources


def test_get_kg_raw_path_for_source_resolver():
    """Test the get_kg_raw_path_for_source resolver function."""

    # Mock pipeline mapping
    mock_mapping = {
        "integration": [
            {"name": "rtx_kg2", "integrate_in_kg": True, "is_private": False, "is_public": True},
            {"name": "robokop", "integrate_in_kg": True, "is_private": False, "is_public": True},
            {"name": "spoke", "integrate_in_kg": True, "is_private": True},
            {"name": "embiology", "integrate_in_kg": True, "is_private": True},
            {"name": "standard_source", "integrate_in_kg": True, "is_private": False},
        ]
    }

    # Mock environment variables
    env_vars = {
        "DEV_GCS_BUCKET": "gs://mtrx-hub-dev-3of",
        "PROD_GCS_BUCKET": "gs://mtrx-us-central1-hub-prod-storage",
        "PUBLIC_GCS_BUCKET": "gs://data.dev.everycure.org",
        "INCLUDE_PRIVATE_DATASETS": "1",  # Simulate prod environment
    }

    with patch("matrix.settings.DYNAMIC_PIPELINES_MAPPING", return_value=mock_mapping), patch.dict(
        os.environ, env_vars
    ):
        # Public sources should use public bucket
        result = get_kg_raw_path_for_source("rtx_kg2")
        assert result == "gs://data.dev.everycure.org/data/01_RAW"

        result = get_kg_raw_path_for_source("robokop")
        assert result == "gs://data.dev.everycure.org/data/01_RAW"

        # Private sources should use prod bucket
        result = get_kg_raw_path_for_source("spoke")
        assert result == "gs://mtrx-us-central1-hub-prod-storage/data/01_RAW"

        result = get_kg_raw_path_for_source("embiology")
        assert result == "gs://mtrx-us-central1-hub-prod-storage/data/01_RAW"

        # Standard sources (neither private nor public) should use dev bucket
        result = get_kg_raw_path_for_source("standard_source")
        assert result == "gs://mtrx-us-central1-hub-dev-storage/data/01_RAW"

        # Unknown source should default to dev bucket
        result = get_kg_raw_path_for_source("unknown_source")
        assert result == "gs://mtrx-us-central1-hub-dev-storage/data/01_RAW"
