from unittest.mock import patch

import pytest
from matrix.git_utils import has_legal_branch_name


@patch("matrix.git_utils.get_current_git_branch")
def test_legal_branch_name_valid(mock_get_branch):
    # Test valid branch names
    valid_branches = ["release/v1.0.0", "release/v2.3.4-alpha", "release/v10.15.20-beta123"]
    for branch in valid_branches:
        mock_get_branch.return_value = branch
        assert has_legal_branch_name()


@patch("matrix.git_utils.get_current_git_branch")
def test_legal_branch_name_invalid(mock_get_branch):
    # Test invalid branch names
    invalid_branches = [
        "/release/v1",
        "release/v1",
        "release/1.0.0",
        "release/v1.0",
        "release/v1.0.0/",
        "v1.0.0",
        "feature/awesome-feature",
        "release/v1.0.0--invalid",
    ]
    for branch in invalid_branches:
        mock_get_branch.return_value = branch
        assert not has_legal_branch_name()
