import subprocess
from typing import List
from unittest.mock import MagicMock, patch

from matrix_cli.commands.code import catchup
from matrix_cli.components.git import get_code_diff, parse_diff_input
from matrix_cli.components.settings import settings


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
    results = [
        subprocess.CompletedProcess(
            args=["mocked"],
            returncode=0,
            stdout="/fake/git/root",
        ),
        subprocess.CompletedProcess(
            args=["mocked"],
            returncode=0,
            stdout="SOMEGITHASH\nANOTHERGITHASH",
        ),
    ]
    mock_subprocess_run.side_effect = results

    # When
    from_ref, to_ref = parse_diff_input("2 weeks ago", "origin/main")

    # Then
    assert mock_subprocess_run.call_count == 2
    assert from_ref == "ANOTHERGITHASH"
    assert to_ref == "origin/main"
    calls = mock_subprocess_run.call_args_list
    assert calls[1][0][0] == ["git", "log", "--since=2 weeks ago", "--format=%H"]
    assert str(calls[1][1]["cwd"]) == "/fake/git/root"


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
    assert mock_subprocess_run.call_count == 3
    args, kwargs = mock_subprocess_run.call_args
    assert args[0][:3] == ["git", "diff", "abc123..def456"]


def test_catchup_command(mock_git_root, mock_subprocess_run, mock_rprint):
    """Test catchup command execution through the complete flow.

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
        # Given
        mock_invoke_model.return_value = "AI generated summary"

        # When
        catchup("1 week ago", "HEAD", disable_rendering=False)

        # Then
        _assert_subprocess_calls(
            mock_subprocess_run,
            [
                ["git", "rev-parse", "--show-toplevel"],
                ["git", "log", "--since=1 week ago", "--format=%H"],
                ["git", "rev-parse", "--show-toplevel"],
                ["git", "diff", "commit1..HEAD", "--", *settings.inclusion_patterns],
            ],
        )
        mock_invoke_model.assert_called_once()
        mock_rprint.assert_called_once()


def _assert_subprocess_calls(mock_subprocess_run, expected_calls: List[List[str]]):
    assert mock_subprocess_run.call_count == len(expected_calls)
    for i in range(len(expected_calls)):
        args = mock_subprocess_run.call_args_list[i].args[0]
        # assert the command is the sam
        assert args == expected_calls[i]
