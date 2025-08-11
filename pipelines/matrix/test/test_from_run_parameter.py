"""Test the --from-run parameter functionality.

This module tests the generic --from-run parameter that allows reading input datasets
from a specified run while writing outputs to the current run.
"""

# NOTE: This test file was partially generated using AI assistance.

import os
import pytest
from unittest.mock import patch, MagicMock

from matrix.cli_commands.run import _set_from_run_environment_variables, _discover_run_based_paths


class TestFromRunParameter:
    """Test the --from-run parameter functionality."""

    def test_discover_run_based_paths_with_mock_session(self):
        """Test discovering run-based paths from catalog configuration."""
        # NOTE: This test was partially generated using AI assistance.
        
        # Mock session and config loader
        mock_session = MagicMock()
        mock_session._conf_source = "/fake/path"
        mock_session._env = "test"
        
        # Mock the config loader to return a paths configuration
        mock_config_loader = MagicMock()
        mock_config_loader.__getitem__.return_value = {
            "paths": {
                "filtering": "data/test/releases/test-release/runs/${run_name}/datasets/filtering",
                "embeddings": "data/test/releases/test-release/runs/${run_name}/datasets/embeddings",
                "modelling": "data/test/releases/test-release/runs/${run_name}/datasets/modelling",
                "evaluation": "data/test/releases/test-release/runs/${run_name}/datasets/evaluation",
                "matrix_generation": "data/test/releases/test-release/runs/${run_name}/datasets/matrix_generation",
                "matrix_transformations": "data/test/releases/test-release/runs/${run_name}/datasets/matrix_transformations",
                "inference": "data/test/releases/test-release/runs/${run_name}/datasets/inference",
                "raw": "data/test/raw",  # This should not be included
                "integration": "data/test/releases/test-release/datasets/integration",  # This should not be included
            }
        }
        
        with patch('matrix.cli_commands.run.settings') as mock_settings:
            mock_settings.CONFIG_LOADER_CLASS.return_value = mock_config_loader
            mock_settings.CONFIG_LOADER_ARGS = {}
            
            result = _discover_run_based_paths(mock_session)
            
            # Verify that only run-based paths are discovered
            expected_paths = [
                "filtering",
                "embeddings", 
                "modelling",
                "evaluation",
                "matrix_generation",
                "matrix_transformations",
                "inference"
            ]
            assert set(result) == set(expected_paths)

    def test_discover_run_based_paths_fallback(self):
        """Test fallback to hardcoded paths when discovery fails."""
        # NOTE: This test was partially generated using AI assistance.
        
        mock_session = MagicMock()
        
        with patch('matrix.cli_commands.run.settings') as mock_settings:
            # Make the config loader raise an exception
            mock_settings.CONFIG_LOADER_CLASS.side_effect = Exception("Config error")
            mock_settings.CONFIG_LOADER_ARGS = {}
            
            result = _discover_run_based_paths(mock_session)
            
            # Should fall back to hardcoded paths
            expected_fallback_paths = [
                "filtering",
                "embeddings", 
                "modelling",
                "evaluation",
                "matrix_generation",
                "matrix_transformations",
                "inference"
            ]
            assert set(result) == set(expected_fallback_paths)

    def test_set_from_run_environment_variables_test_env(self):
        """Test setting environment variables for test environment."""
        # NOTE: This test was partially generated using AI assistance.
        
        mock_session = MagicMock()
        mock_session._conf_source = "/fake/path"
        mock_session._env = "test"
        
        # Mock the config loader to return a paths configuration
        mock_config_loader = MagicMock()
        mock_config_loader.__getitem__.return_value = {
            "paths": {
                "filtering": "data/test/releases/test-release/runs/${run_name}/datasets/filtering",
                "embeddings": "data/test/releases/test-release/runs/${run_name}/datasets/embeddings",
                "modelling": "data/test/releases/test-release/runs/${run_name}/datasets/modelling",
            }
        }
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('matrix.cli_commands.run.settings') as mock_settings:
                mock_settings.CONFIG_LOADER_CLASS.return_value = mock_config_loader
                mock_settings.CONFIG_LOADER_ARGS = {}
                
                _set_from_run_environment_variables("my-old-run", "test", mock_session)
                
                # Verify that environment variables are set correctly for test environment
                assert os.environ["FILTERING"] == "data/test/releases/test-release/runs/my-old-run/datasets/filtering"
                assert os.environ["EMBEDDINGS"] == "data/test/releases/test-release/runs/my-old-run/datasets/embeddings"
                assert os.environ["MODELLING"] == "data/test/releases/test-release/runs/my-old-run/datasets/modelling"

    def test_set_from_run_environment_variables_cloud_env(self):
        """Test setting environment variables for cloud environment."""
        # NOTE: This test was partially generated using AI assistance.
        
        mock_session = MagicMock()
        mock_session._conf_source = "/fake/path"
        mock_session._env = "cloud"
        
        # Mock the config loader to return a paths configuration
        mock_config_loader = MagicMock()
        mock_config_loader.__getitem__.return_value = {
            "paths": {
                "filtering": "${run_dir}/datasets/filtering",
                "embeddings": "${run_dir}/datasets/embeddings",
                "modelling": "${run_dir}/datasets/modelling",
            }
        }
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('matrix.cli_commands.run.settings') as mock_settings:
                mock_settings.CONFIG_LOADER_CLASS.return_value = mock_config_loader
                mock_settings.CONFIG_LOADER_ARGS = {}
                
                _set_from_run_environment_variables("my-old-run", "cloud", mock_session)
                
                # Verify that environment variables are set correctly for cloud environment
                assert os.environ["FILTERING"] == "${run_dir}/datasets/filtering"
                assert os.environ["EMBEDDINGS"] == "${run_dir}/datasets/embeddings"
                assert os.environ["MODELLING"] == "${run_dir}/datasets/modelling"

    def test_set_from_run_environment_variables_fallback(self):
        """Test fallback behavior when config loading fails."""
        # NOTE: This test was partially generated using AI assistance.
        
        mock_session = MagicMock()
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('matrix.cli_commands.run.settings') as mock_settings:
                # Make the config loader raise an exception
                mock_settings.CONFIG_LOADER_CLASS.side_effect = Exception("Config error")
                mock_settings.CONFIG_LOADER_ARGS = {}
                
                _set_from_run_environment_variables("my-old-run", "test", mock_session)
                
                # Should fall back to setting just the run name
                assert os.environ["FILTERING"] == "my-old-run"
                assert os.environ["EMBEDDINGS"] == "my-old-run"
                assert os.environ["MODELLING"] == "my-old-run"

    def test_dynamic_discovery_integration(self):
        """Test the full integration of dynamic discovery with environment variable setting."""
        # NOTE: This test was partially generated using AI assistance.
        
        mock_session = MagicMock()
        mock_session._conf_source = "/fake/path"
        mock_session._env = "test"
        
        # Mock the config loader to return a realistic paths configuration
        mock_config_loader = MagicMock()
        mock_config_loader.__getitem__.return_value = {
            "paths": {
                "filtering": "data/test/releases/test-release/runs/${run_name}/datasets/filtering",
                "embeddings": "data/test/releases/test-release/runs/${run_name}/datasets/embeddings",
                "modelling": "data/test/releases/test-release/runs/${run_name}/datasets/modelling",
                "raw": "data/test/raw",  # Non-run-based path
                "integration": "data/test/releases/test-release/datasets/integration",  # Non-run-based path
            }
        }
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('matrix.cli_commands.run.settings') as mock_settings:
                mock_settings.CONFIG_LOADER_CLASS.return_value = mock_config_loader
                mock_settings.CONFIG_LOADER_ARGS = {}
                
                _set_from_run_environment_variables("my-old-run", "test", mock_session)
                
                # Verify that only run-based paths get environment variables set
                assert "FILTERING" in os.environ
                assert "EMBEDDINGS" in os.environ
                assert "MODELLING" in os.environ
                assert "RAW" not in os.environ  # Should not be set
                assert "INTEGRATION" not in os.environ  # Should not be set
                
                # Verify the correct path structure
                assert os.environ["FILTERING"] == "data/test/releases/test-release/runs/my-old-run/datasets/filtering" 