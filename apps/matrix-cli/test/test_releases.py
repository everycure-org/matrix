# NOTE: This file was partially generated using AI assistance.

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from matrix_cli.components import settings
from matrix_cli.commands.releases import (
    get_pr_details_since,
    get_release_notes,
    suggest_pr_title,
)
from matrix_cli.components.models import PRInfo


@pytest.fixture
def mock_subprocess_run():
    with patch("subprocess.run") as mock:
        mock.return_value.stdout = ""
        mock.return_value.stderr = ""
        yield mock


@pytest.fixture
def mock_get_pr_details():
    with patch("matrix_cli.commands.releases.get_pr_details") as mock:
        yield mock


@pytest.fixture
def mock_vertex_model():
    with patch("vertexai.generative_models.GenerativeModel") as mock:
        model_instance = MagicMock()
        model_instance.generate_content.return_value.text = "AI generated response"
        mock.return_value = model_instance
        yield mock


@pytest.fixture
def mock_vertex_init():
    with patch("vertexai.init") as mock:
        yield mock


def test_get_pr_details_since(mock_subprocess_run, mock_get_pr_details):
    """Test retrieval of PR details since a tag.

    Given:
        - A previous tag
        - Mocked commit logs with PR references
        - Mocked PR details
    When:
        - Getting PR details
    Then:
        - Should return processed PR information
    """
    # Given
    mock_subprocess_run.return_value.stdout = (
        "abc123 Merge pull request #123\ndef456 Fix bug (#456)\nghi789 Related to #789"
    )
    expected_pr_details = pd.DataFrame(
        {
            "number": ["123", "456", "789"],
            "title": ["PR 1", "PR 2", "PR 3"],
        }
    )
    mock_get_pr_details.return_value = expected_pr_details

    # When
    result = get_pr_details_since("v1.0.0", use_cache=True)

    # Then
    assert result.equals(expected_pr_details)
    # Sort the PR numbers to ensure consistent order in comparison
    mock_get_pr_details.assert_called_once_with(sorted([123, 456, 789]), True)


def test_get_release_notes(mock_subprocess_run, mock_get_pr_details, mock_vertex_model, mock_vertex_init, tmp_path):
    """Test generation of release notes.

    Given:
        - A previous tag
        - Mocked PR details and git diff
        - Mocked release template
    When:
        - Generating release notes
    Then:
        - Should return formatted release notes
    """
    # Given
    release_template = """
changelog:
  categories:
    - title: Features
    - title: Bug Fixes
    """
    template_path = tmp_path / ".github" / "release.yml"
    template_path.parent.mkdir(parents=True)
    template_path.write_text(release_template)

    # Mock git commands
    def mock_subprocess_side_effect(*args, **kwargs):
        mock_result = MagicMock()
        if "rev-parse" in args[0]:
            mock_result.stdout = str(tmp_path)
        elif "log" in args[0]:
            mock_result.stdout = "commit1\ncommit2"
        elif "diff" in args[0]:
            mock_result.stdout = "Sample diff output"
        return mock_result

    mock_subprocess_run.side_effect = mock_subprocess_side_effect

    mock_get_pr_details.return_value = pd.DataFrame(
        {
            "number": ["123"],
            "title": ["Test PR"],
        }
    )

    # When
    result = get_release_notes("v1.0.0", "gpt-4")

    # Then
    assert result == "AI generated response"
    mock_vertex_init.assert_called_once()
    mock_vertex_model.assert_called_once_with("gpt-4")


def test_suggest_pr_title(mock_vertex_model, mock_vertex_init):
    """Test AI suggestion for PR titles.

    Given:
        - PR information
        - Example release notes
        - Mocked AI response
    When:
        - Suggesting PR title
    Then:
        - Should return suggested title
    """
    # Given
    pr_info = PRInfo(
        number="123",
        title="Original title",
        current_labels="bug",
        diff="Some code changes",
        url="https://github.com/org/repo/pull/123",
        merge_commit="abc123",
        new_title="",  # Add required field
        new_labels="",  # Add required field
    )
    examples = "Example release notes"
    corrections = "Example corrections"

    # When
    result = suggest_pr_title(pr_info, examples, corrections)

    # Then
    assert result == "AI generated response"
    mock_vertex_init.assert_called_once()
    mock_vertex_model.assert_called_once_with(settings.power_model)


def test_get_release_notes_error_handling(
    mock_subprocess_run, mock_get_pr_details, mock_vertex_model, mock_vertex_init, tmp_path
):
    """Test error handling in release notes generation.

    Given:
        - Invalid git reference
        - Mocked error response
    When:
        - Generating release notes
    Then:
        - Should handle error gracefully
    """
    # Given
    release_template = """
changelog:
  categories:
    - title: Features
    """
    template_path = tmp_path / ".github" / "release.yml"
    template_path.parent.mkdir(parents=True)
    template_path.write_text(release_template)

    # Mock git commands
    def mock_subprocess_side_effect(*args, **kwargs):
        mock_result = MagicMock()
        if "rev-parse" in args[0]:
            mock_result.stdout = str(tmp_path)
        elif "log" in args[0]:
            mock_result.stdout = "commit1\ncommit2"
        elif "diff" in args[0]:
            mock_result.stdout = "Sample diff output"
        return mock_result

    mock_subprocess_run.side_effect = mock_subprocess_side_effect

    mock_get_pr_details.return_value = pd.DataFrame(
        {
            "number": ["123"],
            "title": ["Test PR"],
        }
    )

    mock_vertex_model.return_value.generate_content.side_effect = Exception("API Error")

    # When/Then
    with pytest.raises(Exception) as exc_info:
        get_release_notes("invalid-ref", "gpt-4")
    assert "1" in str(exc_info.value)
