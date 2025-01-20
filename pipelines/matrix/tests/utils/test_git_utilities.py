from unittest.mock import MagicMock, patch

import pytest
from matrix.git_utils import git_tag_exists, has_legal_branch_name


@pytest.mark.parametrize(
    "branch_name",
    [
        "release/v1.0.0",
        "release/v2.3.4-alpha",
        "release/v10.15.20-beta123",
    ],
)
@patch("matrix.git_utils.get_current_git_branch")
def test_legal_branch_name_valid(mock_get_branch, branch_name):
    """Test valid branch names."""
    mock_get_branch.return_value = branch_name
    assert has_legal_branch_name()


@pytest.mark.parametrize(
    "branch_name",
    [
        "/release/v1",
        "release/v1",
        "release/1.0.0",
        "release/v1.0",
        "release/v1.0.0/",
        "v1.0.0",
        "feature/awesome-feature",
        "release/v1.0.0--invalid",
    ],
)
@patch("matrix.git_utils.get_current_git_branch")
def test_legal_branch_name_invalid(mock_get_branch, branch_name):
    """Test invalid branch names."""
    mock_get_branch.return_value = branch_name
    assert not has_legal_branch_name()


@pytest.mark.parametrize(
    "tag, mock_stdout, expected_result",
    [
        ("v1.0.0", "v1.0.0\n", True),
        ("v2.0.0", "", False),
    ],
)
@patch("subprocess.run")
def test_git_tag_exists(mock_subprocess_run, tag, mock_stdout, expected_result):
    mock_result = MagicMock()
    mock_result.stdout = mock_stdout
    mock_subprocess_run.return_value = mock_result
    result = git_tag_exists(tag)
    assert result == expected_result
