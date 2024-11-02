# NOTE: This module was partially generated using AI assistance.

import os
import subprocess
from pathlib import Path

import typer
import yaml
from rich import print
from rich.console import Console
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
console = Console()

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
    disable_rendering: bool = typer.Option(False, help="Disable rendering of the release article"),
):
    """Write a release article for a given git reference."""

    console.print("[green]Collecting release notes...")
    notes = get_release_notes(since, model=model)

    console.print("[green]Collecting previous articles...")
    previous_articles = get_previous_articles()

    console.print("[green]Summarizing code changes...")
    code_summary = get_ai_code_summary(since, model=model)

    # prompt user to give guidance on what to focus on in the release article
    console.print(Markdown(notes))
    focus_direction = console.input(
        "[bold green]Please provide guidance on what to focus on in the release article. Note 'Enter' will end the prompt: "
    )

    prompt = f"""
# Please write a release article based on the following release notes and git diff:

{notes}

# Summary of the code changes:
{code_summary}

# Here are previous release articles for style reference:
{previous_articles}

# Requirements:
- Maintain an objective and professional tone
- Focus on technical accuracy
- Ensure a high signal-to-noise ratio for technical readers

## Focus of the article:
Please focus on the following topics in the release article:
{focus_direction}
        """

    response = invoke_model(prompt, model=model)

    if output_file:
        with open(output_file, "w") as f:
            f.write(response)
        console.print(f"Release article written to: {output_file}")
    elif not disable_rendering:
        console.print("[bold green]Generated release article:")
        console.print("=" * 100)
        console.print(Markdown(response))
        console.print("=" * 100)
    else:
        console.print(response)


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


@app.command(name="release-notes")
def release_notes(
    since: str = typer.Argument(..., help="Git reference (SHA, tag, branch) to summarize from"),
    model: str = typer.Option(settings.base_model, help="Model to use for summarization"),
    output_file: str = typer.Option(None, help="File to write the release notes to"),
):
    """Generate an AI summary of code changes since a specific git reference."""

    try:
        console.print("Generating release notes...")
        response = get_release_notes(since, model)

        if output_file:
            with open(output_file, "w") as f:
                f.write(response)
        else:
            print(Markdown(response))

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}", err=True)
        raise typer.Exit(1)


def get_release_notes(since: str, model: str) -> str:
    release_template = get_release_template()
    console.print("[bold green]Collecting PR details...")
    pr_details_df = get_pr_details_since(since)[["title", "number"]]
    pr_details_dict = pr_details_df.sort_values(by="number").to_dict(orient="records")
    console.print("[bold green]Collecting git diff...")
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
    for file in release_notes_path.glob("**/*.md"):
        with open(file, "r") as f:
            all_articles_content += "=" * 100 + "\n"
            all_articles_content += file.name + "\n"
            all_articles_content += f.read() + "\n\n"
            all_articles_content += "=" * 100 + "\n"

    return all_articles_content


@memory.cache()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=120))
def invoke_model(prompt: str, model: str, generation_config: GenerationConfig = None) -> str:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init()
    model_object = GenerativeModel(model)
    console.print(f"[bold green] Calling Gemini with a prompt of length: {len(prompt)} characters")
    response = model_object.generate_content(prompt, generation_config=generation_config).text
    console.print(f"[bold green] Response received. Total length: {len(response)} characters")
    return response


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
