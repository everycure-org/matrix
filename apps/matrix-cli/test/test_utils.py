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
@patch("matrix_cli.components.utils.get_releases")
def test_get_latest_minor_release(mock_get_releases, releases_list, expected_result):
    mock_get_releases.return_value = releases_list
    result = get_latest_minor_release()
    assert result == expected_result
