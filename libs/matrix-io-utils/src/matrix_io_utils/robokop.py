import polars as pl

SEPARATOR = r"\|"


def robokop_convert_boolean_columns_to_label_columns(nodes_df: pl.DataFrame) -> pl.DataFrame:
    """Convert MONDO_SUPERCLASS_* and CHEBI_ROLE_* to label columns."""

    mondo_superclass_column_names = [x for x in nodes_df.collect_schema().names() if "MONDO_SUPERCLASS" in x]
    chebi_role_column_names = [x for x in nodes_df.collect_schema().names() if "CHEBI_ROLE" in x]

    nodes_df = nodes_df.with_columns(
        [
            pl.lit(None, dtype=pl.Array(pl.String, 1)).alias("MONDO_SUPERCLASSES"),
            pl.lit(None, dtype=pl.Array(pl.String, 1)).alias("CHEBI_ROLES"),
        ]
    ).collect()

    for c in mondo_superclass_column_names:
        fixed_col_name = c.replace("MONDO_SUPERCLASS_", "").replace(":boolean", "")
        nodes_df = nodes_df.with_columns(
            [
                pl.when(pl.col(c).str.contains("true"))
                .then(pl.col("MONDO_SUPERCLASSES").list.concat(pl.lit(fixed_col_name)))
                .otherwise(pl.col("MONDO_SUPERCLASSES"))
                .alias("MONDO_SUPERCLASSES")
            ]
        ).drop(c)

    for c in chebi_role_column_names:
        fixed_col_name = c.replace("CHEBI_ROLE_", "").replace(":boolean", "")
        nodes_df = nodes_df.with_columns(
            [
                pl.when(pl.col(c).str.contains("true"))
                .then(pl.col("CHEBI_ROLES").list.concat(pl.lit(fixed_col_name)))
                .otherwise(pl.col("CHEBI_ROLES"))
                .alias("CHEBI_ROLES")
            ]
        ).drop(c)

    nodes_df = nodes_df.with_columns(
        [
            pl.col("MONDO_SUPERCLASSES").list.join(SEPARATOR).alias("MONDO_SUPERCLASS"),
            pl.col("CHEBI_ROLES").list.join(SEPARATOR).alias("CHEBI_ROLE"),
        ]
    ).drop(["MONDO_SUPERCLASSES", "CHEBI_ROLES"])

    return nodes_df


def robokop_strip_type_from_column_names(input_df: pl.DataFrame) -> pl.DataFrame:
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
    return input_df
