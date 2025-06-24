# %%
from pathlib import Path

import typer
from google.cloud import bigquery
from joblib import Memory
from rich import print as rprint
from rich.table import Table

app = typer.Typer()

SQL_PATH = Path(__file__).parent

# CLI option for caching
default_cache = False

client = None


def get_client():
    global client
    if client is None:
        client = bigquery.Client(project="mtrx-wg2-modeling-dev-9yj")
    return client


def print_result(result: bigquery.QueryJob, block: bool = True):
    # keys of interest are
    keys = [
        "total_bytes_billed",
        "state",
        "errors",
        "transaction_info",
        "total_bytes_processed",
        "num_child_jobs",
        "num_dml_affected_rows",
        "job_id",
        "job_type",
        "project",
        "query_id",
        "query_parameters",
        "query_plan",
        # "exception",
        "errors",
    ]

    data = {}
    for k in keys:
        data[k] = getattr(result, k)
    # print dict as table
    table = Table()
    for k, v in data.items():
        table.add_row(k, str(v))
    if block:
        r = result.result()
        rprint(table)
        return r
    else:
        rprint(table)


# No-op cache decorator
def no_cache(f):
    return f


# Helper to get cache decorator based on flag
def get_cache_decorator(enable_caching: bool):
    if enable_caching:
        memory = Memory(location=SQL_PATH / ".cache", verbose=0)
        return memory.cache
    else:
        return no_cache


def run_query(query: str, enable_caching: bool = False):
    cache_decorator = get_cache_decorator(enable_caching)

    @cache_decorator
    def _run_query_inner(query: str):
        rprint("Sending query to BigQuery")
        client = get_client()
        job = client.query(query)
        job.result()
        print_result(job)
        return job.result().to_dataframe()

    return _run_query_inner(query)


def run_query_for_file(path: Path | str, enable_caching: bool = False):
    with open(path, "r") as f:
        query = f.read()
    return run_query(query, enable_caching=enable_caching)


@app.command()
def create_tables(enable_caching: bool = typer.Option(default_cache, help="Enable caching")):
    """Create schema and tables."""
    rprint("[bold green]Running: create_tables[/bold green]")
    run_query_for_file(SQL_PATH / "01_create_schema_and_tables.sql", enable_caching=enable_caching)


@app.command()
def insert_scores(enable_caching: bool = typer.Option(default_cache, help="Enable caching")):
    """Insert synthetic scores."""
    rprint("[bold green]Running: insert_scores[/bold green]")
    run_query_for_file(SQL_PATH / "02_insert_synthetic_scores.sql", enable_caching=enable_caching)


@app.command()
def create_function(enable_caching: bool = typer.Option(default_cache, help="Enable caching")):
    """Create stability metric function."""
    rprint("[bold green]Running: create_function[/bold green]")
    run_query_for_file(SQL_PATH / "03_create_stability_metric_function.sql", enable_caching=enable_caching)


@app.command()
def run_metrics(enable_caching: bool = typer.Option(default_cache, help="Enable caching")):
    """Run stability metrics and print results."""
    rprint("[bold green]Running: run_metrics[/bold green]")
    df = run_query_for_file(SQL_PATH / "04_run_stability_metrics.sql", enable_caching=enable_caching)
    rprint(df)


@app.command()
def run_all(enable_caching: bool = typer.Option(default_cache, help="Enable caching")):
    """Run the full pipeline: create tables, insert scores, create function, run metrics."""
    create_tables(enable_caching=enable_caching)
    insert_scores(enable_caching=enable_caching)
    create_function(enable_caching=enable_caching)
    run_metrics(enable_caching=enable_caching)


if __name__ == "__main__":
    app()
