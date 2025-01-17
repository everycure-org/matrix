"""
Functions that interact with git locally.
"""

import json
import subprocess
from typing import List, Optional, Tuple

import typer
from matrix_cli.components.cache import memory
from matrix_cli.components.models import PRInfo
from matrix_cli.components.settings import settings
from matrix_cli.components.utils import get_git_root, run_command


@memory.cache
def fetch_pr_detail(pr_number: int) -> PRInfo:
    """
    Fetches PR details including merge commit.
    """
    # Fetch PR details including merge commit
    command = ["gh", "pr", "view", str(pr_number), "--json", "number,title,author,labels,url,mergeCommit,headRefName"]
    pr_json = run_command(command)
    pr_info = json.loads(pr_json)

    # Extract only the login from the author field
    if "author" in pr_info and isinstance(pr_info["author"], dict):
        pr_info["author"] = pr_info["author"].get("login", "unknown")  # Default to "unknown" if login is missing

    # Fetch diff if merge commit exists
    diff = ""
    if merge_commit := pr_info.get("mergeCommit") and pr_info["mergeCommit"].get("oid"):
        try:
            diff = get_code_from_commit(merge_commit)
        except subprocess.CalledProcessError:
            typer.echo(f"\nWarning: Could not fetch diff for PR #{pr_number}", err=True)
    else:
        # If no merge commit, use the head ref name to fetch the diff
        if head_ref_name := pr_info.get("headRefName"):
            try:
                diff = get_code_diff("origin/main", head_ref_name)
            except subprocess.CalledProcessError:
                typer.echo(f"\nWarning: Could not fetch diff for PR #{pr_number}", err=True)

    return PRInfo.from_github_response(pr_info, diff)


def parse_diff_input(since: str, until: str) -> Tuple[str, str]:
    """Parse the diff input to handle both reference ranges and time-based queries.

    Args:
        since: Starting reference or time expression
        until: Ending reference

    Returns:
        Tuple[str, str]: Processed from and to references
    """
    git_root = get_git_root()

    # Check if input looks like a time expression
    if any(time_unit in since.lower() for time_unit in ["day", "week", "month", "year", "ago"]):
        try:
            cmd = ["git", "log", f"--since={since}", "--format=%H"]
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=git_root,
            )
            commits = result.stdout.strip().split("\n")
            if not commits or not commits[-1]:
                raise ValueError(f"No commits found in the specified timeframe: {since}")
            from_ref = commits[-1]  # Get the oldest commit
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to get commits for timeframe '{since}': {e.stderr}")
    else:
        from_ref = since

    return from_ref, until


def get_code_diff(
    since: str, until: str = "origin/main", file_patterns: List[str] = settings.inclusion_patterns
) -> Optional[str]:
    """Get code differences between two git references or time periods.

    Defaults to all files until latest main

    Args:
        since: Starting git reference or time expression
        until: Ending git reference (default: origin/main)

    Returns:
        str: Formatted diff output
    """
    from_ref, to_ref = parse_diff_input(since, until)
    git_root = get_git_root()
    # allow also single ref which gets the diff just for that commit
    diff_ref = f"{from_ref}..{to_ref}" if to_ref is not None else from_ref
    return run_command(["git", "diff", diff_ref, "--", *file_patterns], cwd=git_root)


def get_code_from_commit(commit: str, file_patterns: List[str] = settings.inclusion_patterns) -> Optional[str]:
    git_root = get_git_root()
    command = ["git", "diff", f"{commit}^!", "--", *file_patterns]
    print(command)
    return run_command(command, cwd=git_root)
