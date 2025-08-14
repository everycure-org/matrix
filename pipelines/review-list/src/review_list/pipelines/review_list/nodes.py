import pyspark.sql as ps
from pyspark.sql import Window
from pyspark.sql.functions import col, row_number


def combine_ranked_pair_dataframes(
    weights: dict[str, dict], config: dict[str, any], **dataframes: ps.DataFrame
) -> ps.DataFrame:
    """
    Combine ranked pair dataframes using weighted merge logic.

    Args:
        weights: Parameter structure with weights configuration
        config: Configuration parameters including limit
        **dataframes: Individual dataframes passed as keyword arguments
    """

    # Extract weight values from the parameter structure
    weight_values = {}
    for dataset_name, weight_config in weights.items():
        weight_values[dataset_name] = weight_config.get("weight", 1.0)

    # Get limit from config
    limit = config.get("limit", 1000)

    print(f"Extracted weights: {weight_values}")  # noqa: T201
    print(f"Using limit: {limit}")  # noqa: T201
    print(f"Received dataframes: {list(dataframes.keys())}")  # noqa: T201

    if not dataframes:
        raise ValueError("At least one DataFrame must be provided")

    # Convert to list of (dataframe, weight) tuples for processing
    df_names = list(dataframes.keys())
    dfs_with_weights = []
    
    for df_name in df_names:
        df = dataframes[df_name]
        weight = weight_values.get(df_name, 1.0)
        dfs_with_weights.append((df, weight))

    if len(dfs_with_weights) == 1:
        return dfs_with_weights[0][0]

    # Use the multiple dataframe weighted merge function
    result = weighted_merge_multiple(dfs_with_weights, limit)

    return result


def weighted_merge_multiple(dfs_with_weights, limit):
    """
    Merge any number of Spark DataFrames according to weights and rank order.

    Args:
        dfs_with_weights (list): List of tuples [(df1, weight1), (df2, weight2), ...]
        limit (int): Maximum number of rows in the output.

    Returns:
        Spark DataFrame
    """
    # Ensure weights sum to 1
    total_weight = sum(w for _, w in dfs_with_weights)
    dfs_with_weights = [(df, w / total_weight) for df, w in dfs_with_weights]

    # Calculate quotas for each df
    quotas = [round(limit * w) for _, w in dfs_with_weights]

    # Adjust quotas so sum matches limit (fix rounding issues)
    diff = limit - sum(quotas)
    if diff != 0:
        quotas[0] += diff

    # Window spec for rank ordering
    w_rank = Window.orderBy("rank")

    # Take top N from each DataFrame according to its quota
    selected_dfs = []
    for (df, _), quota in zip(dfs_with_weights, quotas):
        if quota > 0:
            df_ranked = df.withColumn("rn", row_number().over(w_rank))
            selected = df_ranked.filter(col("rn") <= quota).drop("rn")
            selected_dfs.append(selected)

    # Union all
    merged = selected_dfs[0]
    for sdf in selected_dfs[1:]:
        merged = merged.union(sdf)

    # Drop duplicates and sort by rank
    merged = merged.dropDuplicates(["source", "target"]).orderBy("rank").limit(limit)
    
    # Rewrite rank as row_number to ensure no duplicate ranks
    final_rank_window = Window.orderBy("rank")
    merged = merged.withColumn("rank", row_number().over(final_rank_window))

    return merged