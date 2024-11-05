# NOTE: This script was partially generated using AI assistance.

import json
import os
import platform
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, List

import typer
import yaml
from rich import print
from rich.markdown import Markdown
from tqdm.rich import tqdm

from matrix_cli.cache import memory
from matrix_cli.gh_api import get_pr_details, update_prs
from matrix_cli.models import PRInfo
from matrix_cli.modules.code import get_ai_code_summary, get_code_diff
from matrix_cli.settings import settings
from matrix_cli.utils import console, get_git_root, get_markdown_contents, invoke_model, run_command

if TYPE_CHECKING:
    from pandas import pd

app = typer.Typer(
    help="Manage releases and release notes",
    no_args_is_help=True,
)
WORKERS = 8
MODEL = "gpt-4o-mini"


@app.command(name="article")
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
    return get_markdown_contents(release_notes_path)


@app.command("pr-titles")
def prepare_release(
    previous_tag: str,
    output_file: str = None,
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching of PR details"),
    skip_ai: bool = typer.Option(False, "--skip-ai", help="Skip AI title suggestions"),
):
    """
    Prepares release notes by processing PRs merged since the given tag.

    Args:
        previous_tag (str): The previous release git tag.
        output_file (str): The name of the Excel file to create/update.
        no_cache (bool): If True, bypass the cache when fetching PR details.
        skip_ai (bool): If True, skip AI title suggestions.
    """
    typer.echo(f"Collecting PRs since {previous_tag}...")

    if not output_file:
        # use temporary directory
        output_file = tempfile.mktemp(suffix=".xlsx")

    # Get commit logs and extract PR numbers
    pr_details_df = get_pr_details_since(previous_tag, use_cache=not no_cache)

    # Get AI suggestions for titles if not skipped
    if not skip_ai:
        pr_details_df = enhance_pr_titles(pr_details_df)

    # Export to Excel
    write_excel(pr_details_df, output_file)

    typer.echo(f"\nPR details exported to '{output_file}'.")
    typer.echo("\nPlease edit the file with your desired changes:")
    typer.echo("- Review 'ai_suggested_title' column for AI suggestions")
    typer.echo("- Modify 'new_title' column to change PR titles")
    typer.echo("- Modify 'new_labels' column to change PR labels (comma-separated)")

    # Open the file automatically
    open_file_in_desktop_application(output_file)

    # Wait for user to edit the Excel file
    if not typer.confirm("\nHave you edited the file and ready to continue?"):
        typer.echo("Operation cancelled.")
        raise typer.Exit()

    # Read the modified Excel file
    try:
        updated_df = pd.read_excel(
            output_file,
            dtype={
                "number": "str",
                "title": "str",
                "ai_suggested_title": "str",
                "new_title": "str",
                "current_labels": "str",
                "new_labels": "str",
                "url": "str",
                "merge_commit": "str",
            },
            keep_default_na=False,  # This prevents empty cells from becoming NaN
        )
        update_prs(updated_df)
        typer.echo("\nAll PR updates completed!")
    except Exception as e:
        typer.echo(f"Error processing Excel file: {e}", err=True)
        raise typer.Exit(1)


def get_pr_details_since(previous_tag: str, use_cache: bool = True) -> List[PRInfo]:
    commit_messages = get_commit_logs(previous_tag)
    pr_numbers = extract_pr_numbers(commit_messages)
    if not pr_numbers:
        typer.echo("No PRs found since the previous tag.")
        raise typer.Exit(1)
    return get_pr_details(pr_numbers, use_cache)


def get_commit_logs(previous_tag: str) -> List[str]:
    command = ["git", "log", f"{previous_tag}..origin/main", "--oneline"]
    return run_command(command).split("\n")


def extract_pr_numbers(commit_messages: List[str]) -> List[int]:
    pr_numbers = []
    pattern = r"#(\d+)"

    for message in commit_messages:
        matches = re.findall(pattern, message)
        pr_numbers.extend(int(num) for num in matches)

    return list(set(pr_numbers))  # Remove duplicates


def open_file_in_desktop_application(filename: str):
    """
    Opens a file using the default system application.

    Args:
        filename (str): Path to the file to open.
    """
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", filename], check=True)
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", filename], check=True)
        elif platform.system() == "Windows":
            subprocess.run(["start", filename], check=True, shell=True)
        else:
            typer.echo("Unable to open file: Unsupported operating system", err=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error opening file: {e}", err=True)


def load_example_release_notes() -> str:
    """
    Loads the last 3 release notes from GitHub for GPT context.

    Returns:
        str: Concatenated release notes from the last 3 releases
    """

    try:
        # Get list of releases
        releases_json = run_command(["gh", "release", "list", "--json", "name,tagName"])
        releases = json.loads(releases_json)

        # Take first 3 releases
        release_tags = [release["tagName"] for release in releases[:3]]

        # Fetch release notes for each release
        release_notes = []
        for tag in release_tags:
            try:
                notes = run_command(["gh", "release", "view", tag])
                release_notes.append(f"# Release {tag}\n{notes}\n")
            except subprocess.CalledProcessError:
                console.print(f"[yellow]Warning: Could not fetch release notes for {tag}")
                continue

        return "\n".join(release_notes)

    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError) as e:
        console.print(f"[yellow]Warning: Could not fetch release notes: {str(e)}")
        return ""


@memory.cache
def suggest_pr_title(pr_info: PRInfo, examples: str, corrections: str) -> str:
    """
    Uses GPT-4 to suggest an improved PR title based on the PR details and examples.

    Args:
        pr_info (PRInfo): PR information including title, diff, and labels
        examples (str): Example release notes for context
        corrections (str): Examples of corrected AI suggestions

    Returns:
        str: Suggested title
    """
    prompt = f"""You are a technical writer helping to improve PR titles for a release notes document. 
Please suggest a clear, concise title that describes the change's impact and purpose.

Here are examples of good PR titles from previous releases:
{examples}

Here are examples of bad PR titles that an AI generated that we needed to refine:
{corrections}

For this PR:
- Current title: {pr_info.title}
- Labels: {pr_info.current_labels}
- Code changes summary:
{pr_info.diff[:10000]}  # Limit diff size to avoid token limits

Please suggest a new title that follows the style of the examples, focusing on:
1. Clear description of the change
2. Impact on users/developers
3. Technical context when relevant

Respond with ONLY the suggested title, nothing else. Do not wrap the text in quotes.
"""

    try:
        suggested_title = invoke_model(prompt, settings.power_model)
        # log the current title and new title
        typer.echo(f"\nCurrent title: {pr_info.title}")
        typer.echo(f"Proposed new title: {suggested_title}")
        return suggested_title
    except Exception as e:
        typer.echo(f"\nError getting GPT suggestion: {e}", err=True)
        return pr_info.title


def load_corrections() -> str:
    """
    Loads the corrections for bad PR titles.
    """
    corrections_path = Path(__file__).parent / "prompts" / "ai_title_suggestion_corrections.yaml"
    return corrections_path.read_text()


def enhance_pr_titles(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Enhances PR titles using GPT-4 suggestions in parallel.

    Args:
        df (pd.DataFrame): DataFrame containing PR details

    Returns:
        pd.DataFrame: DataFrame with suggested titles
    """
    examples = load_example_release_notes()
    corrections = load_corrections()
    df = df.copy()

    # Function to process a single PR
    def process_pr(idx_pr_info):
        idx, row = idx_pr_info
        pr_info = PRInfo(**row.to_dict())
        suggested_title = suggest_pr_title(pr_info, examples, corrections)
        return idx, suggested_title

    typer.echo("\nGetting AI suggestions for PR titles...")

    # Create list of (index, pr_info) tuples for processing
    pr_items = list(df.iterrows())

    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(8, len(df))  # Limit concurrent API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pr = {executor.submit(process_pr, item): item for item in pr_items}

        # Process completed tasks with progress bar
        for future in tqdm(
            as_completed(future_to_pr), total=len(pr_items), desc="Getting title suggestions", unit="PR"
        ):
            try:
                idx, suggested_title = future.result()
                df.at[idx, "ai_suggested_title"] = suggested_title
                df.at[idx, "new_title"] = df.at[idx, "ai_suggested_title"]  # Keep AI suggestion as default
            except Exception as e:
                typer.echo(f"\nError processing PR: {e}", err=True)

    return df


def write_excel(df: "pd.DataFrame", filename: str):
    """
    Writes DataFrame to Excel, organizing columns for easy review.
    """
    df = df.copy()

    # Reorder columns to put titles together
    columns = [
        "number",
        "title",
        "ai_suggested_title",
        "new_title",
        "current_labels",
        "new_labels",
        "url",
        "merge_commit",
    ]

    # Drop diff column and reorder
    df.drop(columns=["diff"], inplace=True)
    df = df[columns]

    # Add column descriptions
    writer = pd.ExcelWriter(filename, engine="openpyxl")
    df.to_excel(writer, index=False, sheet_name="PRs")

    # Adjust column widths
    worksheet = writer.sheets["PRs"]
    for idx, col in enumerate(df.columns):
        max_length = max(df[col].astype(str).apply(len).max(), len(col))
        worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 100)

    writer.close()


if __name__ == "__main__":
    app()
