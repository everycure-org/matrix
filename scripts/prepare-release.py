# NOTE: This script was partially generated using AI assistance.

import json
import platform
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import openai
import pandas as pd
import typer
from joblib import Memory
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

app = typer.Typer()
memory = Memory(location=".joblib", verbose=0)
WORKERS = 8
MODEL = "gpt-4o-mini"


class PRInfo(BaseModel):
    """Pydantic model for PR information."""

    number: int
    title: str
    current_labels: str = Field(description="Comma-separated list of current labels")
    new_title: str
    new_labels: str = Field(description="Comma-separated list of new labels")
    url: str
    diff: str = ""
    merge_commit: str = Field(default="", description="Git merge commit hash")
    ai_suggested_title: Optional[str] = Field(default=None, description="Title suggested by AI")

    @classmethod
    def from_github_response(cls, pr_info: Dict, diff: str = "") -> "PRInfo":
        """
        Create a PRInfo instance from GitHub API response.

        Args:
            pr_info (Dict): Raw GitHub API response
            diff (str): Git diff content

        Returns:
            PRInfo: Structured PR information
        """
        labels = ",".join([label["name"] for label in pr_info.get("labels", [])])
        return cls(
            number=pr_info["number"],
            title=pr_info["title"],
            current_labels=labels,
            new_title=pr_info["title"],  # Initially same as current title
            new_labels=labels,  # Initially same as current labels
            url=pr_info["url"],
            diff=diff,
            merge_commit=pr_info.get("mergeCommit", {}).get("oid", ""),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding the diff to avoid Excel issues."""
        return self.model_dump()


def run_command(command: List[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error running command {' '.join(command)}: {e.stderr}", err=True)
        raise


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


def fetch_pr_detail_nocache(pr_number: int) -> Optional[PRInfo]:
    """
    Non-cached version of fetch_pr_detail.
    """
    try:
        # Fetch PR details including merge commit
        command = ["gh", "pr", "view", str(pr_number), "--json", "number,title,labels,url,mergeCommit"]
        pr_json = run_command(command)
        pr_info = json.loads(pr_json)

        # Fetch diff if merge commit exists
        diff = ""
        if merge_commit := pr_info.get("mergeCommit", {}).get("oid"):
            try:
                diff = run_command(["git", "show", merge_commit])
            except subprocess.CalledProcessError:
                typer.echo(f"\nWarning: Could not fetch diff for PR #{pr_number}", err=True)

        return PRInfo.from_github_response(pr_info, diff)
    except subprocess.CalledProcessError:
        typer.echo(f"\nWarning: Could not fetch PR #{pr_number}", err=True)
        return None


@memory.cache
def fetch_pr_detail(pr_number: int) -> Optional[PRInfo]:
    """
    Fetches details for a single PR.

    Args:
        pr_number (int): PR number to fetch

    Returns:
        Dict: PR details or None if failed
    """
    return fetch_pr_detail_nocache(pr_number)


def get_pr_details(pr_numbers: List[int], use_cache: bool = True) -> pd.DataFrame:
    """
    Retrieves PR details from GitHub using thread pool.

    Args:
        pr_numbers (List[int]): The list of PR numbers.
        use_cache (bool): Whether to use cached PR details.

    Returns:
        pd.DataFrame: A DataFrame containing PR details.
    """
    fetch_func = fetch_pr_detail if use_cache else fetch_pr_detail_nocache

    # Use number of CPUs or limit to 8 threads to avoid overwhelming the system

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        # Submit all PR fetch tasks
        future_to_pr = {executor.submit(fetch_func, pr_num): pr_num for pr_num in pr_numbers}

        # Process completed tasks with progress bar
        results = []
        for future in tqdm(as_completed(future_to_pr), total=len(pr_numbers), desc="Fetching PR details", unit="PR"):
            pr_info = future.result()
            if pr_info:
                results.append(pr_info.to_dict())

    return pd.DataFrame(results)


def update_prs(df: pd.DataFrame):
    for _, row in tqdm(list(df.iterrows()), desc="Updating PRs", unit="PR"):
        pr_number = row["number"]

        # Update title if changed
        if row["title"] != row["new_title"]:
            try:
                run_command(["gh", "pr", "edit", str(pr_number), "--title", row["new_title"]])
                typer.echo(f"\nUpdated title for PR #{pr_number}")
            except subprocess.CalledProcessError:
                typer.echo(f"\nFailed to update title for PR #{pr_number}", err=True)

        # Update labels if changed

        # FUTURE: Handle labels as well
        if row["current_labels"] != row["new_labels"]:
            current_labels = set(filter(None, str(row["current_labels"]).split(",")))
            new_labels = set(filter(None, str(row["new_labels"]).split(",")))

            labels_to_add = new_labels - current_labels
            labels_to_remove = current_labels - new_labels

            if labels_to_add:
                try:
                    run_command(["gh", "pr", "edit", str(pr_number), "--add-label", ",".join(labels_to_add)])
                    typer.echo(f"\nAdded labels to PR #{pr_number}: {labels_to_add}")
                except subprocess.CalledProcessError:
                    typer.echo(f"\nFailed to add labels to PR #{pr_number}", err=True)

            if labels_to_remove:
                try:
                    run_command(["gh", "pr", "edit", str(pr_number), "--remove-label", ",".join(labels_to_remove)])
                    typer.echo(f"\nRemoved labels from PR #{pr_number}: {labels_to_remove}")
                except subprocess.CalledProcessError:
                    typer.echo(f"\nFailed to remove labels from PR #{pr_number}", err=True)


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
    Loads the example release notes for GPT context.
    """
    example_path = Path(__file__).parent / "release_preparation" / "example_release_notes.txt"
    if example_path.exists():
        return example_path.read_text()
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

    client = openai.OpenAI()
    try:
        response = call_openai(prompt, client)
        suggested_title = response.choices[0].message.content.strip()
        # log the current title and new title
        typer.echo(f"\nCurrent title: {pr_info.title}")
        typer.echo(f"Proposed new title: {suggested_title}")
        return suggested_title
    except Exception as e:
        typer.echo(f"\nError getting GPT suggestion: {e}", err=True)
        return pr_info.title


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=120))
def call_openai(prompt, client):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a technical writer helping to improve PR titles."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=200,
    )

    return response  # Return original title if GPT fails


def load_corrections() -> str:
    """
    Loads the corrections for bad PR titles.
    """
    corrections_path = Path(__file__).parent / "release_preparation" / "ai_title_suggestion_corrections.yaml"
    return corrections_path.read_text()


def enhance_pr_titles(df: pd.DataFrame) -> pd.DataFrame:
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


def write_excel(df: pd.DataFrame, filename: str):
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


@app.command()
def prepare_release(
    previous_tag: str,
    output_file: str = "release_prs.xlsx",
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

    # Get commit logs and extract PR numbers
    commit_messages = get_commit_logs(previous_tag)
    typer.echo(f"Found {len(commit_messages)} commit messages")
    # print top 10
    [typer.echo(f"{msg}") for msg in commit_messages[:10]]
    typer.echo("Extracting PR numbers...")
    pr_numbers = extract_pr_numbers(commit_messages)

    if not pr_numbers:
        typer.echo("No PRs found since the previous tag.")
        raise typer.Exit(1)

    # Get PR details
    pr_details_df = get_pr_details(pr_numbers, use_cache=not no_cache)

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


if __name__ == "__main__":
    app()
