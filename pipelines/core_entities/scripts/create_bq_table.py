import click
from google.cloud import bigquery


@click.command()
@click.option("--project-id", help="Google Cloud project ID")
@click.option("--dataset", help="BigQuery dataset name")
@click.option("--table-name", help="BigQuery table name")
@click.option("--gs-uri", help="Google Cloud Storage URI")
@click.option("--format", "file_format", help="File format (PARQUET, CSV, JSON, etc.)")
@click.option("--delimiter", help="Field delimiter for CSV files (not needed for PARQUET)")
@click.option("--dry-run", is_flag=True, help="Show the SQL query without executing it")
def create_bq_table(
    project_id,
    dataset,
    table_name,
    gs_uri,
    file_format,
    delimiter: str = ",",
    dry_run: bool = False,
):
    """Create an external BigQuery table from a Google Cloud Storage file."""

    client = bigquery.Client(project=project_id)

    # Normalize format to uppercase (BigQuery expects uppercase)
    file_format = file_format.upper() if file_format else None

    # Build OPTIONS dynamically based on format
    options = [f"format = '{file_format}'"]

    # Only include field_delimiter for CSV files
    # Parquet and JSON formats don't use delimiters
    if delimiter and file_format == "CSV":
        options.append(f"field_delimiter = '{delimiter}'")

    options.append(f"uris = ['{gs_uri}']")
    options_str = ",\n  ".join(options)

    create_table_sql = f"""
CREATE OR REPLACE EXTERNAL TABLE `{project_id}.{dataset}.{table_name}`
OPTIONS (
  {options_str}
);
"""

    if dry_run:
        click.echo("SQL Query:")
        click.echo(create_table_sql)
        return

    try:
        click.echo(f"Creating external table {project_id}.{dataset}.{table_name}...")
        job = client.query(create_table_sql)
        job.result()  # Wait for the job to complete
        click.echo("✅ Table created successfully!")
    except Exception as e:
        click.echo(f"❌ Error creating table: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    create_bq_table()
