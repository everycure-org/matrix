from unittest.mock import patch

import pytest

from scripts.bump_version import bump_version


@pytest.mark.parametrize(
    "existing_branches, bump_type, latest_tag, expected_version",
    [
        (["release/0.3.2"], "patch", "v0.3.2", "v0.3.3"),  # No branch conflict
        (["release/v0.3.1"], "patch", "v0.3.0", "v0.3.2"),  # patch conflict
    ],
)
@patch("bump_version.branch_exists")
def test_bump_version(mock_branch_exists, existing_branches, bump_type, latest_tag, expected_version, capsys):
    # Mock branch_exists to return True for existing branches
    mock_branch_exists.side_effect = lambda branch: branch in existing_branches

    # Run the function
    bump_version(bump_type, latest_tag)

    # Capture printed output
    captured = capsys.readouterr()

    # Assert expected release version is printed
    assert f"release_version={expected_version}" in captured.out
