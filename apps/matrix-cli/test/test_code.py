from unittest.mock import MagicMock, patch

import pytest
import typer
from matrix_cli.commands.code import ai_catchup, catchup
from matrix_cli.components.git import get_code_diff, parse_diff_input


def test_parse_diff_input_with_time_expression(mock_git_root, mock_subprocess_run):
    """Test parsing time-based input for git diff.

    Given:
        - A git repository
        - A time-based input expression
    When:
        - Parsing the diff input with a time expression
    Then:
        - Should return the oldest commit and target reference
        - Should call git log with correct parameters
    """
    # Given
    mock_subprocess_run.return_value.stdout = "commit3\ncommit2\ncommit1\n"
    mock_subprocess_run.return_value.returncode = 0

    # When
    from_ref, to_ref = parse_diff_input("2 weeks ago", "origin/main")

    # Then
    assert from_ref == "commit1"
    assert to_ref == "origin/main"
    mock_subprocess_run.assert_called_once()
    args, kwargs = mock_subprocess_run.call_args
    assert args[0] == ["git", "log", "--since=2 weeks ago", "--format=%H"]
    assert str(kwargs["cwd"]) == "/fake/git/root"


def test_get_code_diff_with_valid_inputs(mock_git_root, mock_subprocess_run):
    """Test getting diff output with valid references.

    Given:
        - Two valid git commit references
        - A git repository
    When:
        - Requesting diff between the commits
    Then:
        - Should return the expected diff output
        - Should call git diff with correct parameters
    """
    # Given
    expected_diff = "sample diff output"
    mock_subprocess_run.return_value.stdout = expected_diff
    mock_subprocess_run.return_value.returncode = 0

    # When
    result = get_code_diff("abc123", "def456")

    # Then
    assert result == expected_diff
    mock_subprocess_run.assert_called_once()
    args, kwargs = mock_subprocess_run.call_args
    assert args[0][:3] == ["git", "diff", "abc123..def456"]


def test_ai_catchup_command(mock_git_root, mock_subprocess_run, mock_rprint):
    """Test AI catchup command execution through the complete flow.

    Given:
        - A git repository
        - Mocked git commands responses
        - Mocked AI model
    When:
        - Executing AI catchup command with time-based input
    Then:
        - Should execute git commands in correct sequence
        - Should invoke AI model with diff output
        - Should print the AI-generated summary
    """

    # Given
    def mock_subprocess_side_effect(*args, **kwargs):
        mock_result = MagicMock()
        if "log" in args[0]:
            mock_result.stdout = "commit3\ncommit2\ncommit1\n"
        elif "diff" in args[0]:
            mock_result.stdout = "sample diff output"
        mock_result.returncode = 0
        return mock_result

    mock_subprocess_run.side_effect = mock_subprocess_side_effect

    # When
    with patch("matrix_cli.commands.code.invoke_model") as mock_invoke_model:
        mock_invoke_model.return_value = "AI generated summary"
        ai_catchup("1 week ago", disable_rendering=False)

        # Then
        assert mock_subprocess_run.call_count == 2
        mock_invoke_model.assert_called_once()
        prompt_arg = mock_invoke_model.call_args[0][0]
        assert "sample diff output" in prompt_arg
        mock_rprint.assert_called_once()


def test_catchup_command_error_handling(mock_console):
    """Test error handling in catchup command.

    Given:
        - A mocked console
        - A failing git diff command
    When:
        - Executing catchup command
    Then:
        - Should handle the error gracefully
        - Should display error message
        - Should exit with code 1
    """
    # Given
    error_message = "Git command failed"
    with patch("matrix_cli.commands.code.get_code_diff") as mock_get_diff:
        mock_get_diff.side_effect = ValueError(error_message)

        # When/Then
        with pytest.raises(typer.Exit) as exc_info:
            catchup("abc123")

        assert exc_info.value.exit_code == 1
        mock_console.print.assert_called_once()
        args, _ = mock_console.print.call_args
        assert error_message in args[0]
