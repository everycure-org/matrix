from pathlib import Path

import typer
from rich import print as rprint
from rich.markdown import Markdown

from matrix_cli.components.git import fetch_pr_detail, get_code_diff
from matrix_cli.components.settings import settings
from matrix_cli.components.utils import console, get_git_root, get_markdown_contents, invoke_model

app = typer.Typer(help="Code-related utility commands", no_args_is_help=True)


@app.command()
def catchup(
    since: str = typer.Argument(
        ..., help="Git reference (SHA, tag, branch) to diff from, or time expression (e.g., '2 weeks ago')"
    ),
    until: str = typer.Option(default="origin/main", help="Git reference to diff to (default: origin/main)"),
    model: str = typer.Option(settings.base_model, help="Language Model to use"),
    disable_rendering: bool = typer.Option(False, help="Disable rendering of the summary"),
):
    """Uses AI to summarize code changes since a specific git reference or timeframe."""
    summary = get_ai_code_summary(since, until, model)

    if not disable_rendering:
        rprint(Markdown(summary))
    else:
        console.print(summary)


@app.command()
def branch_summary(model: str = typer.Option(settings.base_model, help="Language Model to use")):
    """Generate an AI summary of the current branch."""
    summary = get_ai_code_summary("origin/main", "HEAD", model)
    rprint(Markdown(summary))


@app.command()
def pr_draft(model: str = typer.Option(settings.base_model, help="Language Model to use")):
    """Generate an AI draft for a PR."""
    summary = get_ai_code_summary("origin/main", "HEAD", model)
    prompt = f"""
    Please prepare a PR description based on the following summary:
    {summary}
    """
    response = invoke_model(prompt, model=model)
    console.print(response)


@app.command()
def pr_summary(
    pr_number: int,
    model: str = typer.Option(settings.base_model, help="Language Model to use"),
    question: str = typer.Option(None, help="Question to ask about the PR"),
    disable_rendering: bool = typer.Option(False, help="Disable rendering of the summary"),
):
    """Generate an AI summary of a specific PR, including code changes and context."""

    console.print(f"[green]Fetching PR #{pr_number} details...")

    try:
        pr_info = fetch_pr_detail(pr_number)
    except Exception as e:
        console.print(f"[bold red]Error: Could not fetch PR #{pr_number}: {e}")
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
