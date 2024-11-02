# NOTE: This module was partially generated using AI assistance.

import os
import subprocess

import typer
from rich import print
from rich.markdown import Markdown

from matrix_cli.cache import memory
from matrix_cli.utils import get_git_root

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


def get_code_diff(since: str) -> str:
    """Get code differences since a specific git reference.

    Args:
        since: Git reference (SHA, tag, or branch) to diff from

    Returns:
        str: Formatted diff output
    """

    try:
        result = subprocess.run(
            ["git", "diff", f"{since}..origin/main", "--", *INCLUSION_PATTERNS],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise typer.BadParameter(f"Failed to get diff: {e.stderr}")


@app.command()
def catchup(
    since: str = typer.Argument(..., help="Git reference (SHA, tag, branch) to diff from"),
):
    """Show code changes since a specific git reference."""
    try:
        diff_output = get_code_diff(since)
        typer.echo(diff_output)
        # print(Markdown(f"```diff\n{diff_output}\n```"))
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def summarize(
    since: str = typer.Argument(..., help="Git reference (SHA, tag, branch) to summarize from"),
    model: str = typer.Option("gemini-1.5-flash-002", help="Model to use for summarization"),
):
    """Generate an AI summary of code changes since a specific git reference."""

    try:
        typer.echo(f"Loading diff...: {model}")
        diff_output = get_code_diff(since)
        typer.echo("Diff loaded! Loading AI model...")
        # Configure Gemini

        prompt = f"""Please provide a concise summary of the following code changes. 
        Focus on creating the content for the following release template:
        {get_release_template()}

        Focus on:
        1. Key functional changes
        2. New features or improvements
        3. Important refactoring
        4. Breaking changes (if any)
        
        Code diff:
        {diff_output}
        """

        response = invoke_model(prompt, model)
        print(Markdown(response))

    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)


def get_release_template() -> str:
    git_root = get_git_root()
    release_template_path = os.path.join(git_root, ".github", "release.yml")
    with open(release_template_path, "r") as f:
        return f.read()


@memory.cache
def invoke_model(prompt: str, model: str) -> str:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    typer.echo(f"Invoking model: {model}")
    vertexai.init()
    model_object = GenerativeModel(model)
    return model_object.generate_content(prompt).text


if __name__ == "__main__":
    app()
