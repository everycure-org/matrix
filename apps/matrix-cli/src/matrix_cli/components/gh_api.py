# NOTE: This script was partially generated using AI assistance.

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, List

import typer
from matrix_cli.components.git import fetch_pr_detail, fetch_pr_detail_nocache
from matrix_cli.components.settings import settings
from matrix_cli.components.utils import console, run_command
from tqdm.rich import tqdm

if TYPE_CHECKING:
    from pandas import pd


def get_pr_details(pr_numbers: List[int], use_cache: bool = True) -> "pd.DataFrame":
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

    with ThreadPoolExecutor(max_workers=settings.workers) as executor:
        # Submit all PR fetch tasks
        future_to_pr = {executor.submit(fetch_func, pr_num): pr_num for pr_num in pr_numbers}

        # Process completed tasks with progress bar
        results = []
        for future in tqdm(as_completed(future_to_pr), total=len(pr_numbers), desc="Fetching PR details", unit="PR"):
            exception = future.exception()
            pr_info = future.result()
            if exception:
                console.print(f"[bold red]Error fetching PR details for PR #{future_to_pr[future]}: {exception}")
            if pr_info:
                results.append(pr_info.to_dict())

    # importing here for CLI performance reasons
    import pandas as pd

    return pd.DataFrame(results)


def update_prs(df: "pd.DataFrame"):
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
