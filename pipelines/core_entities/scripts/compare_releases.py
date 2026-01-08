# Note: this file was generated with AI. It works, but it's not beautiful code.
import os
import re
import subprocess
from datetime import datetime

import click
import pandas as pd


def load_release_file(file_path: str) -> pd.DataFrame:
    """Load a release from a file path."""
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path, keep_default_na=False)
    elif file_path.endswith(".tsv"):
        return pd.read_csv(file_path, sep="\t", keep_default_na=False)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def compare_releases(
    df_new: pd.DataFrame,
    df_base: pd.DataFrame,
    id_column: str = "id",
    name_column: str = "name",
) -> dict[str, pd.DataFrame]:
    """
    Compare two DataFrames and return differences as DataFrames.

    Returns a dictionary with:
    - 'added_columns': DataFrame with one row per added column
    - 'removed_columns': DataFrame with one row per removed column
    - 'added_rows': DataFrame of rows in new but not in base
    - 'removed_rows': DataFrame of rows in base but not in new
    - 'changed_values': DataFrame with column names and change counts per column
    - 'changed_values_examples': DataFrame with up to 5 example changes per column
    """
    # Ensure ID column exists
    if id_column not in df_new.columns or id_column not in df_base.columns:
        raise ValueError(f"ID column '{id_column}' not found in both DataFrames")

    # Set ID as index for easier comparison
    df_new_idx = df_new.set_index(id_column)
    df_base_idx = df_base.set_index(id_column)

    results = {}

    # 1. Column differences
    added_columns = set(df_new.columns) - set(df_base.columns)
    removed_columns = set(df_base.columns) - set(df_new.columns)

    if added_columns:
        results["added_columns"] = pd.DataFrame({"column": sorted(added_columns)})
    else:
        results["added_columns"] = pd.DataFrame({"column": []})

    if removed_columns:
        results["removed_columns"] = pd.DataFrame({"column": sorted(removed_columns)})
    else:
        results["removed_columns"] = pd.DataFrame({"column": []})

    # 2. Row differences (by ID)
    added_row_ids = set(df_new_idx.index) - set(df_base_idx.index)
    removed_row_ids = set(df_base_idx.index) - set(df_new_idx.index)

    if added_row_ids:
        results["added_rows"] = df_new_idx.loc[list(added_row_ids)].reset_index()
    else:
        results["added_rows"] = pd.DataFrame()

    if removed_row_ids:
        results["removed_rows"] = df_base_idx.loc[list(removed_row_ids)].reset_index()
    else:
        results["removed_rows"] = pd.DataFrame()

    # 3. Value changes in common rows and columns
    common_row_ids = set(df_new_idx.index) & set(df_base_idx.index)
    common_columns = (set(df_new.columns) & set(df_base.columns)) - {id_column}

    # Collect all changes and organize by column
    all_changes_by_column = {col: [] for col in common_columns}

    for row_id in common_row_ids:
        row_new = df_new_idx.loc[row_id].to_dict()
        row_base = df_base_idx.loc[row_id].to_dict()

        for col in common_columns:
            val_new = row_new[col]
            val_base = row_base[col]
            # Handle NaN comparisons
            if isinstance(val_new, bool) and isinstance(val_base, bool):
                if pd.isna(val_new) and pd.isna(val_base):
                    continue
                if pd.isna(val_new):
                    val_new = None
                if pd.isna(val_base):
                    val_base = None

            # Compare values
            if str(val_new) != str(val_base):
                change_record = {
                    id_column: row_id,
                    "old_value": val_base,
                    "new_value": val_new,
                }
                # Add name for context if column is not name and name exists
                if col != name_column and name_column in row_new:
                    name_val = row_new.get(name_column)
                    if pd.isna(name_val):
                        name_val = None
                    change_record[name_column] = name_val
                all_changes_by_column[col].append(change_record)

    # Create summary DataFrame with counts per column
    summary_data = []
    examples_data = []

    for col in sorted(common_columns):
        changes = all_changes_by_column[col]
        if changes:
            summary_data.append(
                {
                    "column": col,
                    "change_count": len(changes),
                }
            )

            # Add up to 5 examples per column
            for change in changes[:5]:
                example_record = {
                    "column": col,
                    id_column: change[id_column],
                    "old_value": change["old_value"],
                    "new_value": change["new_value"],
                }
                # Include name if it was captured
                if name_column in change:
                    example_record[name_column] = change[name_column]
                examples_data.append(example_record)

    if summary_data:
        results["changed_values"] = pd.DataFrame(summary_data)
        if examples_data:
            # DataFrame will automatically include all columns present in the data
            # (including name_column when present for non-name column changes)
            results["changed_values_examples"] = pd.DataFrame(examples_data)
        else:
            # Empty DataFrame with base columns
            # Note: If examples_data existed, pandas would automatically include
            # name_column when present in the data for non-name column changes
            cols = ["column", id_column, "old_value", "new_value"]
            results["changed_values_examples"] = pd.DataFrame(columns=cols)
    else:
        results["changed_values"] = pd.DataFrame(columns=["column", "change_count"])
        cols = ["column", id_column, "old_value", "new_value"]
        results["changed_values_examples"] = pd.DataFrame(columns=cols)

    return results


def get_github_repo_url() -> str:
    """Get the GitHub repository URL from GitHub Actions env vars or git remote origin."""
    # First, try GitHub Actions environment variables (available in CI/CD)
    # These don't contain tokens and are the safest option
    github_repo = os.environ.get("GITHUB_REPOSITORY")
    github_server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")

    if github_repo:
        # GITHUB_REPOSITORY format: "owner/repo"
        url = f"{github_server}/{github_repo}"
        return url.rstrip("/")

    # Fallback to git remote (for local development)
    # This may contain tokens in GitHub Actions, so we sanitize them
    cmd = ["git", "remote", "get-url", "origin"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    url = result.stdout.strip()

    if not url:
        raise ValueError("Empty URL from origin remote")

    # Sanitize tokens from URL (e.g., https://x-access-token:TOKEN@github.com/owner/repo.git)  # pragma: allowlist secret
    # Remove any authentication tokens from the URL
    url = re.sub(r"https?://[^@]+@", "https://", url)

    # Convert to GitHub URL format
    # Handle both https and git@ formats
    if url.startswith("git@"):
        url = url.replace("git@github.com:", "https://github.com/")
    # Remove .git suffix if present
    if url.endswith(".git"):
        url = url.rstrip(".git")
    # Remove trailing slash if present
    url = url.rstrip("/")

    return url


def calculate_null_counts(df: pd.DataFrame) -> dict[str, int]:
    """
    Calculate the number of null values per column in a DataFrame.

    Returns a dictionary mapping column names to null counts.
    """
    null_counts = {}
    for col in df.columns:
        null_counts[col] = int(df[col].isna().sum())
    return null_counts


def get_commits_between_tags(base_tag: str, release_tag: str) -> list[dict[str, str]]:
    """
    Get commits between two git tags.

    Returns a list of dictionaries with 'hash' (full), 'message', 'author', 'date'.
    """
    try:
        # Get commits between tags
        cmd = [
            "git",
            "log",
            f"{base_tag}..{release_tag}",
            "--pretty=format:%H|%s|%an|%ad",
            "--date=short",
            "--no-merges",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append(
                    {
                        "hash": parts[0],  # Full hash
                        "message": parts[1],
                        "author": parts[2],
                        "date": parts[3],
                    }
                )

        return commits
    except subprocess.CalledProcessError as e:
        click.echo(f"Warning: Failed to get commits: {e.stderr}", err=True)
        return []
    except FileNotFoundError:
        click.echo("Warning: Git not found. Skipping commit history.", err=True)
        return []


def generate_markdown_report(
    results: dict[str, pd.DataFrame],
    id_column: str = "id",
    name_column: str = "name",
    commits: list[dict[str, str]] | None = None,
    base_tag: str | None = None,
    release_tag: str | None = None,
    github_repo_url: str | None = None,
    base_release_file_path: str | None = None,
    release_file_path: str | None = None,
    null_counts_base: dict[str, int] | None = None,
    null_counts_new: dict[str, int] | None = None,
) -> str:
    """Generate a markdown report from comparison results."""
    lines = ["# Release Comparison Report", ""]

    # Add tag information if provided
    if base_tag and release_tag:
        release_line = f"**New Release:** `{release_tag}`"
        lines.append(release_line)
        lines.append("")

        base_line = f"**Base Release:** `{base_tag}`"
        lines.append(base_line)
        lines.append("")

    lines.append(f"**New Release File:** `{release_file_path}`")
    lines.append("")
    lines.append(f"**Base Release File:** `{base_release_file_path}`")
    lines.append("")

    # 0. Null values per column
    if null_counts_base is not None and null_counts_new is not None:
        lines.append("## Null Values per Column")
        lines.append("")
        lines.append("| Column | Base Release Null Count | New Release Null Count |")
        lines.append("|--------|-------------------------|------------------------|")

        # Get all unique columns from both releases
        all_columns = sorted(set(list(null_counts_base.keys()) + list(null_counts_new.keys())))

        for col in all_columns:
            base_count = null_counts_base.get(col) if col in null_counts_base else None
            new_count = null_counts_new.get(col) if col in null_counts_new else None

            base_count_str = str(base_count) if base_count is not None else "N/A"
            new_count_str = str(new_count) if new_count is not None else "N/A"

            lines.append(f"| `{col}` | {base_count_str} | {new_count_str} |")

        lines.append("")

    # 1. Column changes
    lines.append("## Column Changes")
    lines.append("")

    added_cols = results["added_columns"]
    removed_cols = results["removed_columns"]

    if len(added_cols) > 0:
        lines.append("### Added Columns")
        for col in added_cols["column"]:
            lines.append(f"- `{col}`")
        lines.append("")
    else:
        lines.append("### Added Columns")
        lines.append("*No columns added*")
        lines.append("")

    if len(removed_cols) > 0:
        lines.append("### Removed Columns")
        for col in removed_cols["column"]:
            lines.append(f"- `{col}`")
        lines.append("")
    else:
        lines.append("### Removed Columns")
        lines.append("*No columns removed*")
        lines.append("")

    # 2. Row changes
    lines.append("## Row Changes")
    lines.append("")

    added_rows = results["added_rows"]
    removed_rows = results["removed_rows"]

    lines.append("### Added Rows")
    lines.append(f"**Total:** {len(added_rows)}")
    lines.append("")

    if len(added_rows) > 0:
        lines.append("**Examples (up to 10):**")
        lines.append("")
        lines.append("| ID | Name |")
        lines.append("|----|------|")

        display_rows = added_rows.head(10)
        for _, row in display_rows.iterrows():
            row_id = row.get(id_column, "N/A")
            row_name = row.get(name_column, "N/A") if name_column in row else "N/A"
            lines.append(f"| `{row_id}` | {row_name} |")

    lines.append("")

    lines.append("### Removed Rows")
    lines.append(f"**Total:** {len(removed_rows)}")
    lines.append("")

    if len(removed_rows) > 0:
        lines.append("**Examples (up to 10):**")
        lines.append("")
        lines.append("| ID | Name |")
        lines.append("|----|------|")

        display_rows = removed_rows.head(10)
        for _, row in display_rows.iterrows():
            row_id = row.get(id_column, "N/A")
            row_name = row.get(name_column, "N/A") if name_column in row else "N/A"
            lines.append(f"| `{row_id}` | {row_name} |")

    lines.append("")

    # 3. Value changes
    lines.append("## Value Changes")
    lines.append("")

    changed_values = results["changed_values"]
    changed_examples = results["changed_values_examples"]

    if len(changed_values) > 0:
        lines.append("### Summary by Column")
        lines.append("")
        lines.append("| Column | Number of Changes |")
        lines.append("|--------|-------------------|")
        for _, row in changed_values.iterrows():
            col = row["column"]
            count = row["change_count"]
            lines.append(f"| `{col}` | {count} |")
        lines.append("")

        lines.append("### Examples by Column")
        lines.append("")
        lines.append("*Up to 5 examples per column*")
        lines.append("")

        # Group examples by column
        for _, summary_row in changed_values.iterrows():
            col = summary_row["column"]
            col_examples = changed_examples[changed_examples["column"] == col].head(5)

            if len(col_examples) > 0:
                lines.append(f"#### `{col}`")
                lines.append("")

                # Check if name column is present in examples (for non-name columns)
                has_name = col != name_column and name_column in col_examples.columns

                # Build table header
                if has_name:
                    lines.append("| ID | Name | Old Value | New Value |")
                    lines.append("|----|------|-----------|-----------|")
                else:
                    lines.append("| ID | Old Value | New Value |")
                    lines.append("|----|-----------|-----------|")

                for _, example_row in col_examples.iterrows():
                    ex_id = example_row.get(id_column, "N/A")
                    old_val = example_row["old_value"]
                    new_val = example_row["new_value"]

                    # Truncate long values for readability
                    if old_val is not None and len(str(old_val)) > 100:
                        old_val = str(old_val)[:97] + "..."
                    if new_val is not None and len(str(new_val)) > 100:
                        new_val = str(new_val)[:97] + "..."

                    # Escape pipe characters in markdown tables
                    old_val_str = str(old_val).replace("|", "\\|") if old_val is not None else "*None*"
                    new_val_str = str(new_val).replace("|", "\\|") if new_val is not None else "*None*"

                    # Include name if present
                    if has_name:
                        name_val = example_row.get(name_column, "N/A")
                        if name_val is not None:
                            name_str = str(name_val).replace("|", "\\|")
                            if len(name_str) > 50:
                                name_str = name_str[:47] + "..."
                        else:
                            name_str = "*None*"
                        lines.append(f"| `{ex_id}` | {name_str} | `{old_val_str}` | `{new_val_str}` |")
                    else:
                        lines.append(f"| `{ex_id}` | `{old_val_str}` | `{new_val_str}` |")
                lines.append("")
    else:
        lines.append("*No value changes detected*")
        lines.append("")

    # Add commits section if available
    if commits is not None:
        lines.append("## Commits")
        lines.append("")

        if len(commits) > 0:
            # Get date range and authors
            dates = [commit["date"] for commit in commits if commit.get("date")]
            authors = sorted(set(commit["author"] for commit in commits if commit.get("author")))

            # Format date range
            if dates:
                start_date = min(dates)
                end_date = max(dates)
                # Format date from YYYY-MM-DD to DD/MM/YYYY
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    start_formatted = start_dt.strftime("%d/%m/%Y")
                    end_formatted = end_dt.strftime("%d/%m/%Y")
                except Exception:
                    start_formatted = start_date
                    end_formatted = end_date

                authors_str = ", ".join(authors) if authors else "unknown"
                lines.append(
                    f"{len(commits)} commits between {start_formatted} and {end_formatted} "
                    f"from the following authors: {authors_str}"
                )
            else:
                lines.append(f"{len(commits)} commits")

            lines.append("")

            # Bullet point list of commits with links
            for commit in commits:
                hash_full = commit["hash"]
                hash_short = hash_full[:8]
                message = commit["message"]
                author = commit.get("author", "unknown")
                commit_url = f"{github_repo_url}/commit/{hash_full}"
                lines.append(f"- [{hash_short}]({commit_url}): {message} ({author})")
        else:
            lines.append("*No commits found*")
        lines.append("")

    return "\n".join(lines)


@click.command()
@click.option(
    "--release-file-path",
    "-rf",
    required=True,
    help="File path to the new release (local file or GCS path)",
)
@click.option(
    "--base-release-file-path",
    "-bf",
    required=True,
    help="File path to the base release (local file or GCS path)",
)
@click.option(
    "--release-tag",
    "-rt",
    default=None,
    help="Git tag for the new release (optional, for commit/PR history)",
)
@click.option(
    "--base-release-tag",
    "-bt",
    default=None,
    help="Git tag for the base release (optional, for commit/PR history)",
)
@click.option(
    "--output-markdown",
    "-md",
    default=None,
    help="Path to save the markdown report (optional)",
)
def compare_releases_cli(
    release_file_path: str,
    base_release_file_path: str,
    release_tag: str | None,
    base_release_tag: str | None,
    output_markdown: str | None,
):
    # Compare release files
    click.echo(f"Loading base release: {base_release_file_path}")
    try:
        df_base = load_release_file(base_release_file_path)
    except Exception as e:
        click.echo(f"❌ Failed to load base release: {e}", err=True)
        raise click.Abort()

    click.echo(f"Loading new release: {release_file_path}")
    try:
        df_new = load_release_file(release_file_path)
    except Exception as e:
        click.echo(f"❌ Failed to load new release: {e}", err=True)
        raise click.Abort()

    click.echo("Comparing releases...")
    comparison_results = compare_releases(df_new, df_base, name_column="name")

    click.echo("\nComparison complete. Results stored in DataFrames:")
    click.echo(f"  - Added columns: {len(comparison_results['added_columns'])}")
    click.echo(f"  - Removed columns: {len(comparison_results['removed_columns'])}")
    click.echo(f"  - Added rows: {len(comparison_results['added_rows'])}")
    click.echo(f"  - Removed rows: {len(comparison_results['removed_rows'])}")
    click.echo(f"  - Columns with changed values: {len(comparison_results['changed_values'])}")
    click.echo(f"  - Example changes: {len(comparison_results['changed_values_examples'])}")

    # Calculate null counts for both releases
    click.echo("\nCalculating null value counts...")
    null_counts_base = calculate_null_counts(df_base)
    null_counts_new = calculate_null_counts(df_new)

    # Get commits between tags if tags are provided
    commits = None
    github_repo_url = get_github_repo_url()
    if release_tag and base_release_tag:
        click.echo(f"\nFetching commits between {base_release_tag} and {release_tag}...")
        commits = get_commits_between_tags(base_release_tag, release_tag)
        click.echo(f"  Found {len(commits)} commits")

    # Generate markdown report if requested
    if output_markdown:
        click.echo(f"\nGenerating markdown report: {output_markdown}")
        markdown_content = generate_markdown_report(
            comparison_results,
            commits=commits,
            base_tag=base_release_tag,
            release_tag=release_tag,
            github_repo_url=github_repo_url,
            base_release_file_path=base_release_file_path,
            release_file_path=release_file_path,
            null_counts_base=null_counts_base,
            null_counts_new=null_counts_new,
        )
        with open(output_markdown, "w") as f:
            f.write(markdown_content)
        click.echo(f"✅ Markdown report saved to `{output_markdown}`!")


if __name__ == "__main__":
    compare_releases_cli()
