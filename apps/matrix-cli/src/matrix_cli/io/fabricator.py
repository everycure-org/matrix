"""Misc Fabricator functions for building yaml from real data."""

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, List, Optional

import polars as pl
import ruamel.yaml
import typer

app = typer.Typer()

# https://biolink.github.io/biolink-model/KnowledgeLevelEnum/
# https://biolink.github.io/biolink-model/AgentTypeEnum/
# https://biolink.github.io/biolink-model/DirectionQualifierEnum/
KNOWN_TYPE_VALUES = {
    "agent_type": [
        "manual_agent",
        "automated_agent",
        "data_analysis_pipeline",
        "computational_model",
        "text_mining_agent",
        "image_processing_agent",
        "manual_validation_of_automated_agent",
        "not_provided",
    ],
    "knowledge_level": [
        "knowledge_assertion",
        "logical_entailment",
        "prediction",
        "statistical_association",
        "observation",
        "not_provided",
    ],
    "object_direction_qualifier": ["increased", "upregulated", "decreased", "downregulated"],
    "subject_direction_qualifier": ["increased", "upregulated", "decreased", "downregulated"],
}


def read_sampled_df_tsv(path: Path, limit: int | None = None, select: list[str] | None = None) -> pl.DataFrame:
    """Read a Sampled Dataframe from a TSV file."""
    lf = pl.scan_csv(str(path), separator="\t", has_header=True, ignore_errors=True)
    if select:
        lf = lf.select([pl.col(c) for c in select])
    if limit is not None:
        lf = lf.limit(limit)
    return lf.collect()


@app.command(help="Create KG schema snapshot JSON")
def create_kg_schema_snapshot(
    nodes: Annotated[Path, typer.Option("--nodes", "-n")],
    edges: Annotated[Path, typer.Option("--edges", "-e")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    nodes_prefix_exclusions: Annotated[Optional[List[str]], typer.Option("--nodes-columns-exclusions", "-x")] = None,
    edges_prefix_exclusions: Annotated[Optional[List[str]], typer.Option("--edges-columns-exclusions", "-y")] = None,
    n_snapshot_rows: Annotated[Optional[int], typer.Option("--number_of_snapshot_rows", "-z")] = 100,
) -> None:
    """Create a KG schema snapshot."""
    edges_df = read_sampled_df_tsv(edges, limit=n_snapshot_rows)
    nodes_df = read_sampled_df_tsv(nodes, limit=n_snapshot_rows)

    edges_column_names = filtered_columns(edges_df, edges_prefix_exclusions)
    nodes_column_names = filtered_columns(nodes_df, nodes_prefix_exclusions)

    edges_columns = [build_column_summary(edges_df, cn) for cn in edges_column_names]
    nodes_columns = [build_column_summary(nodes_df, cn) for cn in nodes_column_names]

    profile = {"nodes": nodes_columns, "edges": edges_columns}

    with output.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, sort_keys=False)


@app.command(help="Build YAML from KG schema snapshot")
def build_yaml_from_kg_schema_snapshot(
    nodes: Annotated[Path, typer.Option("--nodes", "-n")],
    edges: Annotated[Path, typer.Option("--edges", "-e")],
    schema_snapshot: Annotated[Path, typer.Option("--schema-snapshot", "-s")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    limit: Annotated[int, typer.Option("--limit", "-l")] = 20,
    rows: Annotated[int, typer.Option("--rows", "-r")] = 100,
) -> None:
    """Build a yaml from a KG schema snapshot."""
    snapshot_content = schema_snapshot.read_text()
    snapshot = json.loads(snapshot_content)

    usable_edges_columns = [c["name"] for c in snapshot.get("edges", [])]
    number_of_edges_to_generate = int(round(limit / 2.0))

    edges_df = read_sampled_df_tsv(edges, limit=number_of_edges_to_generate, select=usable_edges_columns)

    edge_id_columns_df = edges_df.select([pl.col("subject"), pl.col("object")])
    edge_ids_df = pl.concat(
        [
            edge_id_columns_df.select([pl.col("subject").alias("id")]),
            edge_id_columns_df.select([pl.col("object").alias("id")]),
        ]
    ).unique(maintain_order=True)

    selected_edge_ids = [x for x in edge_ids_df.get_column("id").to_list() if x is not None]
    # Build regex: ^(id1|id2|...|idN)$ with escaping special chars
    if selected_edge_ids:
        escaped = [re.escape(x) for x in selected_edge_ids]
        pattern = f"{'|'.join(escaped)}"
    else:
        # No ids; pattern that matches nothing
        pattern = "^(?:)$"

    usable_nodes_columns = [c["name"] for c in snapshot.get("nodes", [])]
    nodes_df = read_sampled_df_tsv(nodes, select=usable_nodes_columns)
    try:
        nodes_df = nodes_df.filter(pl.col("id").str.contains(pattern)).limit(limit)
    except Exception:
        # If id not present, keep empty
        nodes_df = nodes_df.head(0)

    data_map = OrderedDict()
    data_map["nodes"] = create_nodes_map(nodes_df, rows)
    data_map["edges"] = create_edges_map(edges_df, rows)

    with output.open("w", encoding="utf-8") as f:
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = False
        yaml.dump(json.loads(json.dumps(data_map)), f)


@app.command(help="Build YAML from KGX")
def build_yaml_from_kgx(
    nodes: Annotated[Path, typer.Option("--nodes", "-n")],
    edges: Annotated[Path, typer.Option("--edges", "-e")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    limit: Annotated[int, typer.Option("--limit", "-l")] = 20,
    rows: Annotated[int, typer.Option("--rows", "-r")] = 100,
) -> None:
    """Build a yaml from a KGX."""
    size = int(round(limit / 2.0))
    edges_df = read_sampled_df_tsv(edges, limit=size)

    edge_id_columns_df = edges_df.select([pl.col("subject"), pl.col("object")])
    edge_ids_df = pl.concat(
        [
            edge_id_columns_df.select([pl.col("subject").alias("id")]),
            edge_id_columns_df.select([pl.col("object").alias("id")]),
        ]
    ).unique(maintain_order=True)

    selected_edge_ids = [x for x in edge_ids_df.get_column("id").to_list() if x is not None]
    if selected_edge_ids:
        escaped = [re.escape(x) for x in selected_edge_ids]
        pattern = f"^({'|'.join(escaped)})$"
    else:
        pattern = "^(?:)$"

    nodes_df = read_sampled_df_tsv(nodes)
    try:
        nodes_df = nodes_df.filter(pl.col("id").cast(pl.Utf8).str.contains(pattern, literal=False))
    except Exception:
        nodes_df = nodes_df.head(0)

    data_map = OrderedDict()
    data_map["nodes"] = create_nodes_map(nodes_df, rows)
    data_map["edges"] = create_edges_map(edges_df, rows)

    with output.open("w", encoding="utf-8") as f:
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = False
        yaml.dump(json.loads(json.dumps(data_map)), f)


def build_column_summary(df: pl.DataFrame, colname: str) -> dict:
    """Build a summary of a DataFrame column."""
    s = df.get_column(colname)
    datatype = s.dtype

    try:
        s_non_null = s.drop_nulls()
    except Exception:
        s_non_null = s
    try:
        sample = s_non_null.sample(n=6, with_replacement=True, shuffle=True)
    except Exception:
        sample = s_non_null.head()
    if datatype.is_float() or datatype.is_integer():
        values = [int(v) for v in sample.to_list() if v is not None]
    else:
        values = [v for v in sample.to_list() if v is not None]
    # deduplicate while preserving order
    seen = set()
    deduped = []
    for v in values:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return {
        "name": colname,
        "datatype": str(s.dtype),
        "samples": deduped,
    }


def filtered_columns(df: pl.DataFrame, prefixes: list[str] | None) -> list[str]:
    """Determine columns after prefix exclusions."""
    cols = list(df.columns)
    if prefixes:
        cols = [c for c in cols if not any(c.startswith(p) for p in prefixes)]
    return cols


def create_nodes_map(df: pl.DataFrame, rows: int) -> dict:
    """Create a nodes map."""
    # primary columns: exclude those starting with "_"
    primary_columns = [c for c in df.columns if not c.startswith("_")]

    schema = {c: df.get_column(c).dtype for c in df.columns}

    columns_map: OrderedDict[str, dict] = OrderedDict()

    for cn in primary_columns:
        dtype = schema.get(cn)
        column_map: OrderedDict[str, object] = OrderedDict()

        if cn == "id":
            s = df.get_column(cn)
            try:
                s_str = s.cast(pl.Utf8)
            except Exception:
                s_str = s.cast(pl.Utf8, strict=False)
            values = [v for v in s_str.to_list() if v is not None]
            prefixes = []
            seen = set()
            for v in values:
                if ":" in v:
                    pref = v.split(":", 1)[0].replace(".", "_", 1)
                else:
                    pref = ""
                if pref not in seen:
                    seen.add(pref)
                    prefixes.append(pref)
            if not prefixes:
                prefixes = [""]
            column_map["type"] = "generate_unique_id"
            column_map["prefixes"] = prefixes
            column_map["delimiter"] = ":"
        else:
            s = df.get_column(cn)
            # print(f"cn: {cn}, datatype: {dtype}")
            if dtype.is_float() or dtype.is_integer():
                values = [int(v) for v in s.to_list() if v is not None]
                # unique sorted
                uniq = sorted(set(values))
                column_map["type"] = "generate_values"
                column_map["sample_values"] = uniq
            else:
                values = []

                for v in s.to_list():
                    if v is None:
                        continue

                    if str(v).isdigit():
                        values.append(int(v))
                        continue

                    if str(v).replace(".", "", 1).replace("-", "", 1).isdigit():
                        values.append(float(v))
                        continue

                    values.append(str(v).replace(".", "_", 1))

                # unique sorted strings
                uniq = sorted(set(values))
                if not uniq:
                    uniq = [""]
                column_map["type"] = "generate_values"
                column_map["sample_values"] = uniq

        columns_map[cn] = column_map

    return OrderedDict(
        [
            ("columns", columns_map),
            ("num_rows", rows),
        ]
    )


def create_edges_map(df: pl.DataFrame, rows: int) -> dict:
    """Create an edges map."""
    primary_columns = [c for c in df.columns if not c.startswith("_")]

    schema = {c: df.get_column(c).dtype for c in df.columns}

    columns_map: OrderedDict[str, dict] = OrderedDict()

    for cn in primary_columns:
        dtype = schema.get(cn)
        column_map: OrderedDict[str, object] = OrderedDict()

        if cn in ("subject", "object"):
            column_map["type"] = "copy_column"
            column_map["source_column"] = "nodes.id"
            column_map["seed"] = 590590
            column_map["sample"] = OrderedDict([("num_rows", "@edges.num_rows")])
        elif cn in ("agent_type", "knowledge_level", "object_direction_qualifier", "subject_direction_qualifier"):
            sample_values = KNOWN_TYPE_VALUES.get(cn, [])
            column_map["type"] = "generate_values"
            column_map["sample_values"] = sample_values
        else:
            s = df.get_column(cn)
            if dtype == pl.Float64 or dtype == pl.Float32:
                values = [v for v in s.to_list() if v is not None]
                uniq = sorted(set(values))
                column_map["type"] = "generate_values"
                column_map["sample_values"] = uniq
            else:
                values = []
                for v in s.to_list():
                    if v is None:
                        continue

                    if str(v).isdigit():
                        values.append(int(v))
                        continue

                    if str(v).replace(".", "", 1).replace("-", "", 1).isdigit():
                        values.append(float(v))
                        continue

                    values.append(str(v).replace(".", "_", 1))

                uniq = sorted(set(values))
                if not uniq:
                    uniq = [""]
                column_map["type"] = "generate_values"
                column_map["sample_values"] = uniq

        columns_map[cn] = column_map

    return OrderedDict(
        [
            ("columns", columns_map),
            ("num_rows", rows),
        ]
    )


def main() -> None:
    """Wrap the PrimeKG CLI options in a Typer app."""
    app()


if __name__ == "__main__":
    main()
