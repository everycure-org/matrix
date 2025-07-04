#!/usr/bin/env python3
"""Test script to verify the dynamic bucket selection works correctly."""

import os
from unittest.mock import patch

from pipelines.matrix.src.matrix.resolvers import get_bucket_for_source, get_kg_raw_path_for_source


def test_dynamic_bucket_selection():
    """Test the dynamic bucket selection logic."""

    # Mock pipeline mapping reflecting the current settings.py configuration
    mock_mapping = {
        "integration": [
            {"name": "rtx_kg2", "integrate_in_kg": True, "is_private": False, "is_public": True},
            {"name": "robokop", "integrate_in_kg": True, "is_private": False, "is_public": True},
            {"name": "spoke", "integrate_in_kg": True, "is_private": True},
            {"name": "embiology", "integrate_in_kg": True, "is_private": True},
            {"name": "ec_medical_team", "integrate_in_kg": True},  # No flags - should use dev
            {"name": "both_flags", "integrate_in_kg": True, "is_private": True, "is_public": True},
            {"name": "explicit_false", "integrate_in_kg": True, "is_private": False, "is_public": False},
        ]
    }

    # Mock environment variables
    env_vars = {
        "DEV_GCS_BUCKET": "gs://mtrx-hub-dev-3of",
        "PROD_GCS_BUCKET": "gs://mtrx-us-central1-hub-prod-storage",
        "PUBLIC_GCS_BUCKET": "gs://data.dev.everycure.org",
    }

    with patch("matrix.resolvers.DYNAMIC_PIPELINES_MAPPING", return_value=mock_mapping), patch.dict(
        os.environ, env_vars, clear=False
    ):
        test_cases = [
            ("rtx_kg2", "gs://data.dev.everycure.org/data/01_RAW", "Public (is_public=True)"),
            ("robokop", "gs://data.dev.everycure.org/data/01_RAW", "Public (is_public=True)"),
            ("spoke", "gs://mtrx-us-central1-hub-prod-storage/data/01_RAW", "Prod (is_private=True)"),
            ("embiology", "gs://mtrx-us-central1-hub-prod-storage/data/01_RAW", "Prod (is_private=True)"),
            ("ec_medical_team", "gs://mtrx-hub-dev-3of/data/01_RAW", "Dev (no flags)"),
            ("unknown_source", "gs://mtrx-hub-dev-3of/data/01_RAW", "Dev (not found)"),
            ("both_flags", "gs://data.dev.everycure.org/data/01_RAW", "Both flags True (public takes precedence)"),
            ("explicit_false", "gs://mtrx-hub-dev-3of/data/01_RAW", "Both flags False (dev)"),
            ("", "gs://mtrx-hub-dev-3of/data/01_RAW", "Empty string as source name"),
            (None, "gs://mtrx-hub-dev-3of/data/01_RAW", "None as source name"),
            ("RTX_KG2", "gs://mtrx-hub-dev-3of/data/01_RAW", "Case sensitivity (should not match)"),
        ]

        for source_name, expected_path, description in test_cases:
            if source_name is None:
                result = get_kg_raw_path_for_source(None)
            else:
                result = get_kg_raw_path_for_source(source_name)
            assert (
                result == expected_path
            ), f"Failed for {source_name}: expected {expected_path}, got {result} ({description})"
            # Path formatting check
            assert result.endswith("/data/01_RAW"), f"Path formatting failed for {source_name}"

    # Edge: missing environment variables
    with patch("matrix.resolvers.DYNAMIC_PIPELINES_MAPPING", return_value=mock_mapping), patch.dict(
        os.environ, {"DEV_GCS_BUCKET": "", "PROD_GCS_BUCKET": "", "PUBLIC_GCS_BUCKET": ""}, clear=True
    ):
        result = get_kg_raw_path_for_source("rtx_kg2")
        assert result.endswith("/data/01_RAW"), "Path formatting with empty env vars should still end with /data/01_RAW"

    # Edge: empty mapping
    with patch("matrix.resolvers.DYNAMIC_PIPELINES_MAPPING", return_value={}), patch.dict(
        os.environ, env_vars, clear=False
    ):
        result = get_kg_raw_path_for_source("rtx_kg2")
        assert result == "gs://mtrx-hub-dev-3of/data/01_RAW", "Empty mapping should fallback to dev bucket"

    print("Testing legacy get_bucket_for_source function:")
    with patch("matrix.resolvers.DYNAMIC_PIPELINES_MAPPING", return_value=mock_mapping):
        dev_bucket = "gs://mtrx-hub-dev-3of"
        public_bucket = "gs://data.dev.everycure.org"

        legacy_test_cases = [
            ("rtx_kg2", public_bucket, "Public (is_public=True)"),
            ("robokop", public_bucket, "Public (is_public=True)"),
            ("spoke", dev_bucket, "Dev (is_private=True, but legacy function only handles public)"),
            ("embiology", dev_bucket, "Dev (is_private=True, but legacy function only handles public)"),
            ("ec_medical_team", dev_bucket, "Dev (no flags)"),
            ("both_flags", public_bucket, "Both flags True (public takes precedence)"),
            ("explicit_false", dev_bucket, "Both flags False (dev)"),
            ("", dev_bucket, "Empty string as source name"),
            (None, dev_bucket, "None as source name"),
        ]

        for source_name, expected_bucket, description in legacy_test_cases:
            if source_name is None:
                result = get_bucket_for_source(None, dev_bucket, public_bucket)
            else:
                result = get_bucket_for_source(source_name, dev_bucket, public_bucket)
            assert (
                result == expected_bucket
            ), f"Failed for {source_name}: expected {expected_bucket}, got {result} ({description})"

        # Both buckets the same
        same_bucket = "gs://same-bucket"
        result = get_bucket_for_source("rtx_kg2", same_bucket, same_bucket)
        assert result == same_bucket, "Legacy: both buckets same should return the same bucket"
