# NOTE: This script was partially generated using AI assistance.

import json
import re
import subprocess
from typing import List

import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer()


def run_command(command: List[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error running command {' '.join(command)}: {e.stderr}", err=True)
        raise


def get_commit_logs(previous_tag: str) -> List[str]:
    command = ["git", "log", f"{previous_tag}..HEAD", "--oneline"]
    return run_command(command).split("\n")


def extract_pr_numbers(commit_messages: List[str]) -> List[int]:
    pr_numbers = []
    pattern = r"#(\d+)"

    for message in commit_messages:
        matches = re.findall(pattern, message)
        pr_numbers.extend(int(num) for num in matches)

    return list(set(pr_numbers))  # Remove duplicates


def get_pr_details(pr_numbers: List[int]) -> pd.DataFrame:
    pr_data = []

    for pr_number in tqdm(pr_numbers, desc="Fetching PR details", unit="PR"):
        try:
            command = ["gh", "pr", "view", str(pr_number), "--json", "number,title,labels"]
            pr_json = run_command(command)
            pr_info = json.loads(pr_json)

            # Extract labels as a comma-separated string
            labels = ",".join([label["name"] for label in pr_info.get("labels", [])])

            pr_data.append(
                {
                    "number": pr_info["number"],
                    "title": pr_info["title"],
                    "current_labels": labels,
                    "new_title": pr_info["title"],  # Column for edited title
                    "new_labels": labels,  # Column for edited labels
                }
            )

        except subprocess.CalledProcessError:
            typer.echo(f"\nWarning: Could not fetch PR #{pr_number}", err=True)
            continue

    return pd.DataFrame(pr_data)


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
        if row["current_labels"] != row["new_labels"]:
            current_labels = set(filter(None, row["current_labels"].split(",")))
            new_labels = set(filter(None, row["new_labels"].split(",")))

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


@app.command()
def prepare_release(previous_tag: str, output_file: str = "release_prs.xlsx"):
    typer.echo(f"Collecting PRs since {previous_tag}...")

    # Get commit logs and extract PR numbers
    commit_messages = get_commit_logs(previous_tag)
    pr_numbers = extract_pr_numbers(commit_messages)

    if not pr_numbers:
        typer.echo("No PRs found since the previous tag.")
        raise typer.Exit(1)

    # Get PR details and export to Excel
    pr_details_df = get_pr_details(pr_numbers)
    pr_details_df.to_excel(output_file, index=False)

    typer.echo(f"\nPR details exported to '{output_file}'.")
    typer.echo("\nPlease edit the file with your desired changes:")
    typer.echo("- Modify 'new_title' column to change PR titles")
    typer.echo("- Modify 'new_labels' column to change PR labels (comma-separated)")

    # Wait for user to edit the Excel file
    if not typer.confirm("\nHave you edited the file and ready to continue?"):
        typer.echo("Operation cancelled.")
        raise typer.Exit()

    # Read the modified Excel file
    try:
        updated_df = pd.read_excel(output_file)
        update_prs(updated_df)
        typer.echo("\nAll PR updates completed!")
    except Exception as e:
        typer.echo(f"Error processing Excel file: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
