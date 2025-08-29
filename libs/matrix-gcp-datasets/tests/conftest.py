# NOTE: This file was partially generated using AI assistance.
"""Shared test fixtures for matrix-gcp-datasets library."""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def kedro_session():
    """Create a mock Kedro session for testing."""

    # For library testing, we don't need a full Kedro session
    # Just return a mock object that has the necessary attributes
    class MockKedroSession:
        pass

    return MockKedroSession()


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return SparkSession.builder.appName("matrix-gcp-datasets-test").getOrCreate()
