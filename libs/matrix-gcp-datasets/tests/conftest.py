# NOTE: This file was partially generated using AI assistance.
"""Shared test fixtures for matrix-gcp-datasets library."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def kedro_session():
    """Create a mock Kedro session for testing."""

    # For library testing, we don't need a full Kedro session
    # Just return a mock object that has the necessary attributes
    class MockKedroContext:
        """Mock Kedro context for testing."""

        def __init__(self):
            self.env = "test"
            self.project_path = Path.cwd()
            self.config_loader = MagicMock()
            # Provide minimal Spark config for testing
            self.config_loader.__getitem__.return_value = {
                "spark.app.name": "test-app",
                "spark.master": "local[*]",
            }

    class MockKedroSession:
        """Mock Kedro session for testing."""

        def __init__(self):
            self._context = MockKedroContext()

        def _get_context(self):
            return self._context

    return MockKedroSession()


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return SparkSession.builder.appName("matrix-gcp-datasets-test").getOrCreate()
