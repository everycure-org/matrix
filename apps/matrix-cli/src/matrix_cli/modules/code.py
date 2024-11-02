# NOTE: This module was partially generated using AI assistance.

import subprocess

import typer
from rich import print
from rich.markdown import Markdown

from matrix_cli.settings import settings
from matrix_cli.utils import console, get_git_root, invoke_model

app = typer.Typer(help="Code-related utility commands")

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
    since: str = typer.Argument(..., help="Git reference (SHA, tag, branch) to diff from"),
    model: str = typer.Option(settings.base_model, help="Model to use for release article generation"),
):
    """Show code changes since a specific git reference."""
    try:
        diff_output = get_code_diff(since)
        console.print(diff_output)
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command(name="ai-catchup")
def ai_catchup(
    since: str = typer.Argument(..., help="Git reference (SHA, tag, branch) to diff from"),
    model: str = typer.Option(settings.base_model, help="Model to use for release article generation"),
):
    """Show code changes since a specific git reference."""

    summary = get_ai_code_summary(since, model)
    print(Markdown(summary))


def get_ai_code_summary(since: str, model: str):
    """Show code changes since a specific git reference."""
    diff_output = get_code_diff(since)

    prompt = f"""
    Please provide a structured and detailed summary of the following code changes.
    Please quote select key code snippets in the diff using code blocks if they are 
    relevant to a developer using the larger project. 

    Code diff:
    {diff_output}
    """

    return invoke_model(prompt, model=model)


def get_code_diff(since: str) -> str:
    """Get code differences since a specific git reference.

    Args:
        since: Git reference (SHA, tag, or branch) to diff from

    Returns:
        str: Formatted diff output
    """
    try:
        # Get the git root directory
        git_root = get_git_root()

        # Run the git diff command from the root directory
        result = subprocess.run(
            ["git", "diff", f"{since}..origin/main", "--", *INCLUSION_PATTERNS],
            check=True,
            capture_output=True,
            text=True,
            cwd=git_root,  # Set the working directory to git root
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise console.print(f"[bold red]Failed to get diff: {e.stderr}", err=True)
