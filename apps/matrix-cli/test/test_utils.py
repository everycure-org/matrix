from unittest.mock import MagicMock, patch

import pytest
from matrix_cli.components.utils import get_latest_minor_release


@pytest.mark.parametrize(
    "releases_list, expected_result",
    [
        (["v0.1.2", "v0.2.0", "v0.3.0", "v0.3.3", "v0.3.5"], "v0.3.0"),
        (["v0.1", "v0.2", "v0.2.5"], "v0.2"),
    ],
)
def test_get_latest_minor_release(releases_list, expected_result):
    result = get_latest_minor_release(releases_list)
    assert result == expected_result
