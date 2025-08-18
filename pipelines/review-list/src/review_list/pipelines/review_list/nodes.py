import logging
import math
from functools import reduce

import pyspark.sql as ps
from pyspark.sql import Window
from pyspark.sql.functions import col, row_number

logger = logging.getLogger(__name__)

# NOTE: This file was partially generated using AI assistance.

def prefetch_top_quota(
    weights: dict[str, dict],
    config: dict[str, any],
    **dataframes: ps.DataFrame,
) -> list[ps.DataFrame]:
    """
    Return the top rows from each input DataFrame according to its quota with a 20% buffer.

    - Quota is computed as round(limit * weight) per dataset, adjusted so the sum equals limit
    - We then take ceil(quota * 1.2) rows ordered by rank (20% buffer)

    Args:
        weights: Mapping of dataset name to a dict containing a 'weight' float
        config: Mapping containing 'limit' int
        **dataframes: Named input DataFrames keyed by dataset name

    Returns:
        List[DataFrame]: A list of trimmed DataFrames in the same order as inputs
    """
    if not dataframes:
        raise ValueError("At least one DataFrame must be provided")

    if "limit" not in config:
        raise ValueError("Missing limit in config")
    limit = config["limit"]

    total_weight = sum(w["weight"] for _, w in weights.items())
    if total_weight != 1:
        raise ValueError("Weights must sum to 1")

    # Compute quotas per dataset, adjust rounding diff onto first dataset
    dataset_names = list(dataframes.keys())
    raw_quotas = [round(limit * weights[name]["weight"]) for name in dataset_names]

    # Apply 20% buffer and select top rows by rank

    trimmed_dataframes: list[ps.DataFrame] = []
    for name, quota in zip(dataset_names, raw_quotas):
        buffer_n = int(math.ceil(quota * 1.2))
        top_plus_buffer = dataframes[name].filter(col("rank") <= buffer_n)
        trimmed_dataframes.append(top_plus_buffer)

    breakpoint() 
    return trimmed_dataframes


# # TODO: maybe use inject_object()
# def combine_ranked_pair_dataframes(
#     weights: dict[str, dict], config: dict[str, any], **dataframes: ps.DataFrame
# ) -> ps.DataFrame:
#     """
#     Combine ranked pair dataframes using weighted merge logic.

#     Args:
#         weights: Parameter structure with weights configuration
#         config: Configuration parameters including limit
#         **dataframes: Individual dataframes passed as keyword arguments
#     """
#     if not dataframes:
#         raise ValueError("At least one DataFrame must be provided")

#     limit = config["limit"]
#     weight_values = {}
#     for dataset_name, weight_config in weights.items():
#         # TODO: handle all edge cases
#         if "weight" not in weight_config:
#             raise ValueError(f"Missing weight for dataset {dataset_name}")
#         weight_values[dataset_name] = weight_config["weight"]

#     dfs_with_weights = [
#         (dataframes[name], weight_values[name]) for name in dataframes.keys()
#     ]

#     return weighted_merge_multiple(dfs_with_weights, limit)


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


def combine_ranked_pair_dataframes(
    weights: dict[str, dict], config: dict[str, any], **trimmed_dataframes: ps.DataFrame
) -> ps.DataFrame:
    """
    Combine multiple ranked DataFrames according to weights and limit.

    Args:
        weights: Mapping of dataset name to a dict containing a 'weight' float
        config: Mapping containing 'limit' int
        **trimmed_dataframes: Named trimmed DataFrames keyed by dataset name

    Returns:
        DataFrame: Combined and deduplicated DataFrame with sequential ranks
    """
    if not trimmed_dataframes:
        raise ValueError("At least one DataFrame must be provided")

    if "limit" not in config:
        raise ValueError("Missing limit in config")
    limit = config["limit"]

    # Extract weight values in the same order as trimmed_dataframes
    dataset_names = list(trimmed_dataframes.keys())
    weight_values = [weights[name]["weight"] for name in dataset_names]
    
    # Ensure weights sum to 1 (with small tolerance)
    total_weight = sum(weight_values)
    if abs(total_weight - 1.0) > 1e-10:
        raise ValueError("Weights must sum to 1")

    # Create list of (dataframe, weight) tuples
    dfs_with_weights = [(trimmed_dataframes[name], weight) for name, weight in zip(dataset_names, weight_values)]

    # Shortcut for single dataframe
    if len(dfs_with_weights) == 1:
        return dfs_with_weights[0][0].limit(limit)

    return weighted_merge_multiple(dfs_with_weights, limit)
