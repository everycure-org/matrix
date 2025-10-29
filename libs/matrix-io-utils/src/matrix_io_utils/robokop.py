import logging

import polars as pl

logger = logging.getLogger(__name__)

SEPARATOR = r"\|"


def robokop_convert_boolean_columns_to_label_columns(nodes_df: pl.LazyFrame) -> pl.DataFrame:
    """Convert MONDO_SUPERCLASS_* and CHEBI_ROLE_* to label columns."""

    mondo_superclass_column_names = [x for x in nodes_df.collect_schema().names() if "MONDO_SUPERCLASS" in x]
    chebi_role_column_names = [x for x in nodes_df.collect_schema().names() if "CHEBI_ROLE" in x]

    ms_df = nodes_df.with_columns(
        [
            pl.lit(None, dtype=pl.Array(pl.String, 1)).alias("MONDO_SUPERCLASSES"),
        ]
    ).select([pl.col("id"), pl.col("category"), pl.col("MONDO_SUPERCLASSES"), pl.col("^MONDO_SUPERCLASS_.*$")])

    cr_df = nodes_df.with_columns(
        [
            pl.lit(None, dtype=pl.Array(pl.String, 1)).alias("CHEBI_ROLES"),
        ]
    ).select([pl.col("id"), pl.col("category"), pl.col("CHEBI_ROLES"), pl.col("^CHEBI_ROLE_.*$")])

    nodes_df = nodes_df.drop(mondo_superclass_column_names).drop(chebi_role_column_names).collect()

    ms_df = _build_label_and_drop_bool(mondo_superclass_column_names, ms_df, "MONDO_SUPERCLASS_", "MONDO_SUPERCLASSES")
    nodes_df = nodes_df.join(ms_df, on=["id", "category"], how="full", coalesce=True)

    cr_df = _build_label_and_drop_bool(chebi_role_column_names, cr_df, "CHEBI_ROLE_", "CHEBI_ROLES")
    nodes_df = nodes_df.join(cr_df, on=["id", "category"], how="full", coalesce=True)

    nodes_df = nodes_df.with_columns(
        [
            pl.col("MONDO_SUPERCLASSES").list.join(SEPARATOR).alias("MONDO_SUPERCLASS"),
            pl.col("CHEBI_ROLES").list.join(SEPARATOR).alias("CHEBI_ROLE"),
        ]
    ).drop(["MONDO_SUPERCLASSES", "CHEBI_ROLES"])

    return nodes_df


def _build_label_and_drop_bool(
    column_names: list[str], df: pl.LazyFrame, bool_col_prefix: str, bool_col_label: str
) -> pl.DataFrame:
    for c in column_names:
        fixed_col_name = c.replace(bool_col_prefix, "").replace(":boolean", "")
        df = df.with_columns(
            [
                pl.when(pl.col(c).str.contains("true"))
                .then(pl.col(bool_col_label).list.concat(pl.lit(fixed_col_name)))
                .otherwise(pl.col(bool_col_label))
                .alias(bool_col_label)
            ]
        )

    df = df.drop(column_names).collect()
    return df


def robokop_strip_type_from_column_names(input_df: pl.LazyFrame) -> pl.DataFrame:
    """Strip the type from the column names."""

    suffixes = [":string[]", ":string", ":float[]", ":float", ":int[]", ":int", ":boolean"]
    rename_map = {}

    # Build the map in a single pass
    for c in input_df.collect_schema().names():
        new_name = c
        for suffix in suffixes:
            if c.endswith(suffix):
                new_name = c.removesuffix(suffix)
                break

        if new_name != c:
            rename_map[c] = new_name

    if rename_map:
        input_df = input_df.rename(rename_map)
    return input_df.collect()
