#!/usr/bin/env python3
"""Test to verify that dynamic bucket selection works for all configured sources."""

import os
from unittest.mock import patch

from matrix.resolvers import get_kg_raw_path_for_source

env_vars = {
    "DEV_GCS_BUCKET": "gs://mtrx-hub-dev-3of",
    "PROD_GCS_BUCKET": "gs://mtrx-us-central1-hub-prod-storage",
    "PUBLIC_GCS_BUCKET": "gs://data.dev.everycure.org",
}


def test_get_kg_raw_path_for_source_resolves_correct_bucket_based_on_flags():
    """Test that the dynamic system works correctly for all sources."""

    # Mock the current configuration from settings.py
    mock_mapping = {
        "integration": [
            {"name": "rtx_kg2", "integrate_in_kg": True, "is_private": False, "is_public": True},
            {"name": "spoke", "integrate_in_kg": True, "is_private": True},
            {"name": "embiology", "integrate_in_kg": True, "is_private": True},
            {"name": "robokop", "integrate_in_kg": True, "is_private": False, "is_public": True},
            {"name": "ec_medical_team", "integrate_in_kg": True},  # no flags = dev bucket
            {"name": "drug_list", "integrate_in_kg": False, "has_edges": False},
            {"name": "disease_list", "integrate_in_kg": False, "has_edges": False},
            {"name": "ground_truth", "integrate_in_kg": False, "has_nodes": False},
            {"name": "ec_clinical_trails", "integrate_in_kg": False},
            {"name": "off_label", "integrate_in_kg": False, "has_nodes": False},
        ]
    }

    with patch("matrix.settings.DYNAMIC_PIPELINES_MAPPING", return_value=mock_mapping), patch.dict(
        os.environ, env_vars
    ):
        # Test cases that should use PUBLIC bucket
        public_sources = [x["name"] for x in mock_mapping["integration"] if x.get("is_public", False)]
        expected = "gs://data.dev.everycure.org/data/01_RAW"
        for source in public_sources:
            result = get_kg_raw_path_for_source(source)
            assert result == expected

        # Test cases that should use PROD bucket
        private_sources = [x["name"] for x in mock_mapping["integration"] if x.get("is_private", False)]
        expected = "gs://mtrx-us-central1-hub-prod-storage/data/01_RAW"
        for source in private_sources:
            result = get_kg_raw_path_for_source(source)
            assert result == expected

        # Test cases that should use DEV bucket (default)
        dev_sources = [
            x["name"]
            for x in mock_mapping["integration"]
            if not x.get("is_private", False) and not x.get("is_public", False)
        ]
        expected = "gs://mtrx-hub-dev-3of/data/01_RAW"
        for source in dev_sources:
            result = get_kg_raw_path_for_source(source)
            assert result == expected

        # Test adding a new source with is_public flag
        new_mapping = {
            "integration": mock_mapping["integration"]
            + [
                {"name": "new_public_source", "integrate_in_kg": True, "is_public": True},
                {"name": "new_private_source", "integrate_in_kg": True, "is_private": True},
                {"name": "new_default_source", "integrate_in_kg": True},
            ]
        }

        with patch("matrix.settings.DYNAMIC_PIPELINES_MAPPING", return_value=new_mapping):
            # New public source should automatically use public bucket
            result = get_kg_raw_path_for_source("new_public_source")
            expected = "gs://data.dev.everycure.org/data/01_RAW"
            assert result == expected
            # New private source should automatically use prod bucket
            result = get_kg_raw_path_for_source("new_private_source")
            expected = "gs://mtrx-us-central1-hub-prod-storage/data/01_RAW"
            assert result == expected

            # New default source should automatically use dev bucket
            result = get_kg_raw_path_for_source("new_default_source")
            expected = "gs://mtrx-hub-dev-3of/data/01_RAW"
            assert result == expected
