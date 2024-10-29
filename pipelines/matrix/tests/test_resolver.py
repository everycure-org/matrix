from matrix.resolvers import env
import pytest
import os
from typing import Generator


@pytest.fixture
def _mock_env_variables() -> Generator[None, None, None]:
    original_env = os.environ.copy()
    os.environ["TEST_KEY"] = "test_value"
    yield
    os.environ.clear()
    os.environ.update(original_env)


def test_env_existing_key(_mock_env_variables: None) -> None:
    val = env("TEST_KEY")
    assert val == "test_value"


def test_env_non_existent_key(_mock_env_variables: None) -> None:
    with pytest.raises(KeyError):
        env("NON_EXISTENT_KEY")

    val = env("FOOBAR", "default_value")
    assert val == "default_value"
