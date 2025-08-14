import pyspark.sql as ps


def combine_ranked_pair_dataframes(
    weights: dict[str, dict], **dataframes: ps.DataFrame
) -> ps.DataFrame:
    """
    DUMMY FUNCTION - real logic not implemented yet!
    Will be replaced with review list ranking logic.

    Args:
        weights: Parameter structure with weights configuration
        **dataframes: Individual dataframes passed as keyword arguments
    """

    # Extract weight values from the parameter structure
    weight_values = {}
    for catalog_entry, config in weights.items():
        base_name = catalog_entry.split("@")[0]
        weight_values[base_name] = config.get("weight", 1.0)

    # Ignore T201 ruff
    print(f"Extracted weights: {weight_values}")  # noqa: T201
    print(f"Received dataframes: {list(dataframes.keys())}")  # noqa: T201

    if not dataframes:
        raise ValueError("At least one DataFrame must be provided")
    # Convert dataframes dict to list for processing
    df_list = list(dataframes.values())
    combined = df_list[0]
    for df in df_list[1:]:
        combined = combined.union(df)

    return combined
