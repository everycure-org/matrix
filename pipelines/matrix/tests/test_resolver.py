from matrix.resolvers import env
import pytest
import os


@pytest.fixture
def mock_env_variables():
    original_env = os.environ.copy()
    os.environ["TEST_KEY"] = "test_value"
    yield
    os.environ.clear()
    os.environ.update(original_env)


def test_env_existing_key(mock_env_variables):
    val = env("TEST_KEY")
    assert val == "test_value"


def test_env_non_existent_key(mock_env_variables):
    with pytest.raises(KeyError):
        env("NON_EXISTENT_KEY")

    val = env("FOOBAR", "default_value")
    assert val == "default_value"
