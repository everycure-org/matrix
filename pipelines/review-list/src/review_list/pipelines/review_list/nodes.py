import logging
from functools import reduce

import pyspark.sql as ps
from pyspark.sql import Window
from pyspark.sql.functions import col, row_number

logger = logging.getLogger(__name__)


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
    if total_weight != 1:
        raise ValueError("Weights must sum to 1")

    # Calculate quota - number of rows for each dataframe
    quotas = [round(limit * w) for _, w in dfs_with_weights]

    # Adjust quota of first dataframe so sum matches limit (fix rounding issues)
    diff = limit - sum(quotas)
    if diff != 0:
        quotas[0] += diff

    # Take top N from each DataFrame according to its quota
    selected_rows_from_dfs = []
    for i, ((df, weight), quota) in enumerate(zip(dfs_with_weights, quotas)):
        if quota > 0:
            df_count = df.count()
            if df_count < quota:
                logger.warning(
                    f"WARNING: DataFrame {i+1} has only {df_count} rows but quota is {quota}. Taking all available rows."
                )
            df_ranked = df.withColumn("rn", row_number().over(Window.orderBy("rank")))
            selected = df_ranked.filter(col("rn") <= quota).drop("rn")
            selected_rows_from_dfs.append(selected)

    # Union all dataframes
    merged = reduce(lambda x, y: x.union(y), selected_rows_from_dfs)
    merged = merged.dropDuplicates(["source", "target"]).orderBy("rank").limit(limit)

    # TODO: Optional - recalculate the quotas based on deduplicated data

    # Rewrite rank as row_number to ensure no duplicate ranks
    merged = merged.withColumn("rank", row_number().over(Window.orderBy("rank")))

    return merged
