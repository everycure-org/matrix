"""
Pytest configuration and shared fixtures for the test suite.
"""

import os
import random
import string

import pytest


@pytest.fixture(scope="session")
def integration_test_setup():
    """
    Session-scoped fixture for integration test setup and cleanup.

    Yields:
        dict: Repository configuration

    The fixture handles cleanup of the created repository automatically.
    """
    # Generate 3-5 character random suffix
    suffix_length = random.randint(3, 5)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=suffix_length))

    # Use environment variable or default
    base_repo = os.getenv("HF_INTEGRATION_REPO", "everycure/kedro-datasets-integration-tests")
    repo_id = f"{base_repo}-{suffix}"

    repo_config = {
        "repo_id": repo_id,
        "suffix": suffix,
        "base_repo": base_repo,
        "private": True,  # Always use private repos for integration tests
    }

    yield repo_config

    # Cleanup the created repository
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.delete_repo(repo_id, repo_type="dataset")
        print(f"Cleaned up repository: {repo_id}")
    except Exception as e:
        print(f"Failed to cleanup repository {repo_id}: {e}")
