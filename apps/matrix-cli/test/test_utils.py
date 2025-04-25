import pytest

from matrix_cli.components.utils import get_latest_minor_release


@pytest.mark.parametrize(
    "releases_list, expected_result",
    [
        (["v0.1", "v0.2", "v0.2.5"], "v0.2"),
        (["v2.1.2", "v1.2.3", "v1.3.2", "v1.1.2", "v1.1.1"], "v2.1.2"),
        (["v2.1.5", "v2.1.2", "v1.2.3", "v1.3.2", "v1.1.2", "v1.1.1"], "v2.1.2"),
    ],
)
def test_get_latest_minor_release(releases_list, expected_result):
    result = get_latest_minor_release(releases_list)
    assert result == expected_result
