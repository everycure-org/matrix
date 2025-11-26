"""Matrix CLI IO."""

from pathlib import Path

import polars as pl


def read_tsv_lazy(path: Path, delimiter: str = ",") -> pl.LazyFrame:
    """Read TSV to a LazyFrame."""
    return pl.scan_csv(
        path,
        separator=delimiter,
        infer_schema_length=0,
        low_memory=True,
        has_header=True,
        ignore_errors=True,
        truncate_ragged_lines=True,
    )
