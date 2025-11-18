import logging

import polars as pl

logger = logging.getLogger(__name__)

SEPARATOR = "|"


def robokop_convert_boolean_columns_to_label_columns(nodes_df: pl.LazyFrame) -> pl.DataFrame:
    """Convert MONDO_SUPERCLASS_* and CHEBI_ROLE_* to label columns."""

    mondo_superclass_column_names = [x for x in nodes_df.collect_schema().names() if "MONDO_SUPERCLASS" in x]
    chebi_role_column_names = [x for x in nodes_df.collect_schema().names() if "CHEBI_ROLE" in x]

    ms_df = nodes_df.with_columns(
        [
            pl.lit([], dtype=pl.List(pl.String)).alias("MONDO_SUPERCLASSES"),
        ]
    ).select([pl.col("id"), pl.col("category"), pl.col("MONDO_SUPERCLASSES"), pl.col("^MONDO_SUPERCLASS_.*$")])

    cr_df = nodes_df.with_columns(
        [
            pl.lit([], dtype=pl.List(pl.String)).alias("CHEBI_ROLES"),
        ]
    ).select([pl.col("id"), pl.col("category"), pl.col("CHEBI_ROLES"), pl.col("^CHEBI_ROLE_.*$")])

    nodes_df = nodes_df.drop(mondo_superclass_column_names).drop(chebi_role_column_names).collect()

    cr_df = _build_label_and_drop_bool(chebi_role_column_names, cr_df, "CHEBI_ROLE_", "CHEBI_ROLES")
    nodes_df = (
        nodes_df.join(cr_df, on=["id", "category"], how="full", coalesce=True)
        .with_columns(pl.col("CHEBI_ROLES").list.join(SEPARATOR).alias("CHEBI_ROLE"))
        .drop(["CHEBI_ROLES"])
    )

    ms_df = _build_label_and_drop_bool(mondo_superclass_column_names, ms_df, "MONDO_SUPERCLASS_", "MONDO_SUPERCLASSES")
    nodes_df = (
        nodes_df.join(ms_df, on=["id", "category"], how="full", coalesce=True)
        .with_columns(pl.col("MONDO_SUPERCLASSES").list.join(SEPARATOR).alias("MONDO_SUPERCLASS"))
        .drop(["MONDO_SUPERCLASSES"])
    )

    return nodes_df


def _build_label_and_drop_bool(
    column_names: list[str], df: pl.LazyFrame, bool_column_prefix: str, bool_column_label: str
) -> pl.DataFrame:
    for column_name in column_names:
        fixed_column_name = column_name.replace(bool_column_prefix, "").replace(":boolean", "")
        df = df.with_columns(
            [
                pl.when(pl.col(column_name).str.contains(pl.lit("true")))
                .then(pl.col(bool_column_label).list.concat(pl.lit(fixed_column_name)))
                .otherwise(pl.col(bool_column_label))
                .alias(bool_column_label)
            ]
        )

    df = df.drop(column_names).collect()
    return df


def robokop_strip_type_from_column_names(input_df: pl.LazyFrame) -> pl.DataFrame:
    """Strip the type from the column names."""

    suffixes = [":string[]", ":string", ":float[]", ":float", ":int[]", ":int", ":boolean"]
    rename_map = {}

    # Build the map in a single pass
    for column_name in input_df.collect_schema().names():
        new_name = column_name
        for suffix in suffixes:
            if column_name.endswith(suffix):
                new_name = column_name.removesuffix(suffix)
                break

        if new_name != column_name:
            rename_map[column_name] = new_name

    if rename_map:
        input_df = input_df.rename(rename_map)
    return input_df.collect()
