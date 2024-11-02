# NOTE: This module was partially generated using AI assistance.

import os
import subprocess
from pathlib import Path

import typer
import yaml
from rich import print
from rich.markdown import Markdown
from tenacity import retry, stop_after_attempt, wait_exponential
from vertexai.generative_models import (
    GenerationConfig,
)

from matrix_cli.cache import memory
from matrix_cli.modules.releases import get_pr_details_since
from matrix_cli.settings import settings
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


@app.command(name="release-article")
def write_release_article(
    since: str = typer.Argument(..., help="Git reference (SHA, tag, branch) to diff from"),
    output_file: str = typer.Option(None, help="File to write the release article to"),
    model: str = typer.Option(settings.power_model, help="Model to use for release article generation"),
):
    """Write a release article for a given git reference."""

    typer.echo("Collecting release notes...")
    notes = _get_release_notes(since, model=settings.base_model)
    typer.echo("Collecting previous articles...")
    previous_articles = get_previous_articles()
    typer.echo("Summarizing code changes...")
    code_summary = ai_code_summary(since, model=settings.base_model)

    prompt = f"""Please write a release article based on the following release notes and git diff:
    {notes}

    Summary of the code changes:
    {code_summary}


    Here are previous release articles to help with style and tone:
    {previous_articles}
    """

    typer.echo(f"Invoking model with a prompt of length: {len(prompt)} characters")

    generation_config = GenerationConfig(max_output_tokens=10_000)
    response = invoke_model(prompt, model=model, generation_config=generation_config)

    if output_file:
        with open(output_file, "w") as f:
            f.write(response)
    else:
        print(response)


@app.command()
def catchup(
    since: str = typer.Argument(..., help="Git reference (SHA, tag, branch) to diff from"),
    model: str = typer.Option(settings.base_model, help="Model to use for release article generation"),
):
    """Show code changes since a specific git reference."""
    try:
        diff_output = get_code_diff(since)
        typer.echo(diff_output)
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command(name="ai-catchup")
def ai_catchup(
    since: str = typer.Argument(..., help="Git reference (SHA, tag, branch) to diff from"),
    model: str = typer.Option(settings.base_model, help="Model to use for release article generation"),
):
    """Show code changes since a specific git reference."""

    summary = ai_code_summary(since, model)
    print(Markdown(summary))


def ai_code_summary(since: str, model: str):
    """Show code changes since a specific git reference."""
    diff_output = get_code_diff(since)
    prompt = f"""
    Please provide a structured and detailed summary of the following code changes.

    Code diff:
    {diff_output}
    """

    return invoke_model(prompt, model=model)


@app.command(name="release-notes")
def release_notes(
    since: str = typer.Argument(..., help="Git reference (SHA, tag, branch) to summarize from"),
    model: str = typer.Option(settings.base_model, help="Model to use for summarization"),
    output_file: str = typer.Option(None, help="File to write the release notes to"),
):
    """Generate an AI summary of code changes since a specific git reference."""

    try:
        typer.echo("Generating release notes...")
        response = _get_release_notes(since, model)

        if output_file:
            with open(output_file, "w") as f:
                f.write(response)
        else:
            print(Markdown(response))

    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)


def _get_release_notes(since: str, model: str) -> str:
    release_template = get_release_template()
    typer.echo("Collecting PR details...")
    pr_details_df = get_pr_details_since(since)[["title", "number"]]
    pr_details_dict = pr_details_df.to_dict(orient="records")
    typer.echo("Collecting git diff...")
    diff_output = get_code_diff(since)

    prompt = f"""Please provide a concise summary of the following code changes. 
    Focus on creating the content for the following release template following its categories:

    ```yaml
    {release_template}
    ```

    Code diff:

    ```diff
    {diff_output}
    ```

    PR details:

    ```yaml
    {yaml.dump(pr_details_dict)}
    ```

    """

    typer.echo(f"Invoking model with a prompt of length: {len(prompt)} characters")
    return invoke_model(prompt, model)


def get_release_template() -> str:
    git_root = get_git_root()
    release_template_path = os.path.join(git_root, ".github", "release.yml")
    with open(release_template_path, "r") as f:
        return f.read()


def get_previous_articles() -> list[str]:
    git_root = get_git_root()
    release_notes_path = Path(git_root) / "docs" / "src" / "releases" / "posts"
    # load all markdown files in the directory
    all_articles_content = ""
    for file in release_notes_path.glob("*.md"):
        with open(file, "r") as f:
            all_articles_content += f.read() + "\n\n"

    return all_articles_content


@memory.cache
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=120))
def invoke_model(prompt: str, model: str, **kwargs) -> str:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init()
    model_object = GenerativeModel(model)
    return model_object.generate_content(prompt, **kwargs).text


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
        raise typer.BadParameter(f"Failed to get diff: {e.stderr}")
