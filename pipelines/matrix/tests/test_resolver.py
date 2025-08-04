import os

import pytest
from matrix.resolvers import env
from matrix_auth.environment import load_environment_variables


@pytest.fixture
def mock_env_variables():
    original_env = os.environ.copy()
    os.environ["TEST_KEY"] = "test_value"
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_env_files(tmp_path):
    """Create temporary .env and .env.defaults files for testing."""
    original_cwd = os.getcwd()

    defaults_file = tmp_path / ".env.defaults"
    defaults_file.write_text("SHARED_KEY=default_value\nOVERRIDE_KEY=default_value")

    env_file = tmp_path / ".env"
    env_file.write_text("OVERRIDE_KEY=custom_value\nLOCAL_KEY=local_value")

    # Change to temp directory so find_dotenv works correctly
    os.chdir(tmp_path)
    yield

    # Cleanup
    os.chdir(original_cwd)


def test_env_existing_key(mock_env_variables):
    val = env("TEST_KEY")
    assert val == "test_value"


def test_env_non_existent_key(mock_env_variables):
    with pytest.raises(KeyError):
        env("NON_EXISTENT_KEY")

    val = env("FOOBAR", "default_value")
    assert val == "default_value"


def test_env_file_precedence(mock_env_files, mock_env_variables):
    """Test that .env properly overrides .env.defaults values."""
    load_environment_variables()

    # Should get value from .env.defaults
    assert env("SHARED_KEY") == "default_value"

    # Should get overridden value from .env
    assert env("OVERRIDE_KEY") == "custom_value"

    # Should get local-only value from .env
    assert env("LOCAL_KEY") == "local_value"
