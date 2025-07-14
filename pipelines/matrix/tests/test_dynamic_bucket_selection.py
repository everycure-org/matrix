#!/usr/bin/env python3
"""Test script to verify the dynamic bucket selection works correctly."""

import os
from unittest.mock import patch

from matrix.resolvers import get_kg_raw_path_for_source


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
            ("rtx_kg2", f"${env_vars['PUBLIC_GCS_BUCKET']}/data/01_RAW", "Public (is_public=True)"),
            ("robokop", f"${env_vars['PUBLIC_GCS_BUCKET']}/data/01_RAW", "Public (is_public=True)"),
            ("spoke", f"${env_vars['PROD_GCS_BUCKET']}/data/01_RAW", "Prod (is_private=True)"),
            ("embiology", f"${env_vars['PROD_GCS_BUCKET']}/data/01_RAW", "Prod (is_private=True)"),
            ("ec_medical_team", f"${env_vars['DEV_GCS_BUCKET']}/data/01_RAW", "Dev (no flags)"),
            ("unknown_source", f"${env_vars['DEV_GCS_BUCKET']}/data/01_RAW", "Dev (not found)"),
            (
                "both_flags",
                f"${env_vars['PUBLIC_GCS_BUCKET']}/data/01_RAW",
                "Both flags True (public takes precedence)",
            ),
            ("explicit_false", f"${env_vars['DEV_GCS_BUCKET']}/data/01_RAW", "Both flags False (dev)"),
            ("", f"${env_vars['DEV_GCS_BUCKET']}/data/01_RAW", "Empty string as source name"),
            (None, f"${env_vars['DEV_GCS_BUCKET']}/data/01_RAW", "None as source name"),
            ("RTX_KG2", f"${env_vars['DEV_GCS_BUCKET']}/data/01_RAW", "Case sensitivity (should not match)"),
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
        assert result == f"${env_vars['DEV_GCS_BUCKET']}/data/01_RAW", "Empty mapping should fallback to dev bucket"
