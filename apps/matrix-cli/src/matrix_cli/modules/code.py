# NOTE: This module was partially generated using AI assistance.

import subprocess
from pathlib import Path
from typing import Optional, Tuple

import typer
from rich import print as rprint
from rich.markdown import Markdown

from matrix_cli.gh_api import fetch_pr_detail_nocache
from matrix_cli.settings import settings
from matrix_cli.utils import console, get_git_root, get_markdown_contents, invoke_model

app = typer.Typer(help="Code-related utility commands", no_args_is_help=True)

# use git diff patterns here
INCLUSION_PATTERNS = [
    "*.md",
    "*.py",
    "*.yaml",
    "*.tf",
    "*.Makefile",
    "*.Dockerfile",
    "*.sh",
    "*.toml",
    "*.yml",
    "*.txt",
    "*.hcl",
    "*.git",
    ":!**/matrix/packages/*",
]


@app.command()
def catchup(
    since: str = typer.Argument(
        ..., help="Git reference (SHA, tag, branch) to diff from, or time expression (e.g., '2 weeks ago')"
    ),
    until: str = typer.Option("origin/main", help="Git reference to diff to (default: origin/main)"),
    model: str = typer.Option(settings.base_model, help="Model to use for release article generation"),
    disable_rendering: bool = typer.Option(False, help="Disable rendering of the code diff"),
):
    """Show code changes since a specific git reference or timeframe."""
    try:
        diff = get_code_diff(since, until)
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}")
        raise typer.Exit(1)

    if not disable_rendering:
        rprint(Markdown(diff))
    else:
        console.print(diff)


@app.command(name="ai-catchup")
def ai_catchup(
    since: str = typer.Argument(
        ..., help="Git reference (SHA, tag, branch) to diff from, or time expression (e.g., '2 weeks ago')"
    ),
    until: str = typer.Option("origin/main", help="Git reference to diff to (default: origin/main)"),
    model: str = typer.Option(settings.base_model, help="Model to use for release article generation"),
    disable_rendering: bool = typer.Option(False, help="Disable rendering of the summary"),
):
    """Uses AI to summarize code changes since a specific git reference or timeframe."""
    summary = get_ai_code_summary(since, until, model)

    if not disable_rendering:
        rprint(Markdown(summary))
    else:
        console.print(summary)


@app.command()
def pr_summary(
    pr_number: int,
    model: str = typer.Option(settings.base_model, help="Model to use for PR summary generation"),
    question: str = typer.Option(None, help="Question to ask about the PR"),
    disable_rendering: bool = typer.Option(False, help="Disable rendering of the summary"),
):
    """Generate an AI summary of a specific PR, including code changes and context."""

    console.print(f"[green]Fetching PR #{pr_number} details...")

    # Use existing utility to fetch PR details
    pr_info = fetch_pr_detail_nocache(pr_number)
    if not pr_info:
        console.print(f"[bold red]Error: Could not fetch PR #{pr_number}")
        raise typer.Exit(1)

    prompt = f"""Please provide a comprehensive summary of this Pull Request.
    Focus on the technical changes and their impact.

    PR Title: {pr_info.title}
    PR Labels: {pr_info.current_labels}
    
    Code Changes:
    ```diff
    {pr_info.diff}
    ```

    Please structure the summary with the following sections:
    1. Overview of Changes
    2. Technical Implementation Details
    3. Potential Impact and Considerations
    4. Testing Approach (if evident from the changes)

    Keep the tone technical and focus on the key changes and their implications.

    Optionally, the user may have submitted the following question:
    {question}
    """

    console.print("[green]Generating AI summary...")
    response = invoke_model(prompt, model=model)

    if not disable_rendering:
        console.print("\n[bold green]PR Summary:")
        console.print("=" * 80)
        console.print(Markdown(response))
        console.print("=" * 80)
    else:
        console.print(response)


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


def get_code_diff(since: str, until: str = "origin/main") -> Optional[str]:
    """Get code differences between two git references or time periods.

    Args:
        since: Starting git reference or time expression
        until: Ending git reference (default: origin/main)

    Returns:
        str: Formatted diff output
    """
    git_root = get_git_root()
    from_ref, to_ref = parse_diff_input(since, until)

    command = ["git", "diff", f"{from_ref}..{to_ref}", "--", *INCLUSION_PATTERNS]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=git_root,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to get diff: {e.stderr}")


def get_ai_code_summary(since: str, until: str = "origin/main", model: str = settings.base_model):
    """Show code changes between two git references or time periods."""
    diff_output = get_code_diff(since, until)

    prompt = f"""
    Please provide a structured and detailed summary of the following code changes.
    Please quote select key code snippets in the diff using code blocks if they are 
    relevant to a developer using the larger project. 

    Structure the summary in a way that it is "top down" and focuses on the structural changes,
    rather than the individual lines of code. 

    To help you understand the changes in a bigger picture, here is the content of our onboarding guide
    (which is not part of the diff):

    {get_markdown_contents(Path(get_git_root()) / "docs" / "src" / "onboarding")}

    ACTUAL CODE DIFF: 

    {diff_output}
    """

    return invoke_model(prompt, model=model)
