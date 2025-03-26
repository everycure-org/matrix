from unittest.mock import patch

import pytest

from scripts.bump_version import bump_version


@pytest.mark.parametrize(
    "existing_branches, bump_type, latest_tag, expected_version",
    [
        (["release/v0.3.2"], "patch", "v0.3.2", "v0.3.3"),  # no branch conflict
        (["release/v0.3.1"], "patch", "v0.3.0", "v0.3.2"),  # patch conflict
        (["release/v0.3.1", "release/v0.3.2"], "patch", "v0.3.0", "v0.3.3"),  # patch conflict
        (["release/v0.4.0"], "minor", "v0.3.0", "v0.5.0"),  # minor conflict
    ],
)
@patch("scripts.bump_version.branch_exists")
def test_bump_version(mock_branch_exists, existing_branches, bump_type, latest_tag, expected_version):
    mock_branch_exists.side_effect = lambda branch: branch in existing_branches
    result = bump_version(bump_type, latest_tag)

    assert result == expected_version
