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

    # Convert to list for processing
    df_names = list(dataframes.keys())
    df_list = list(dataframes.values())

    if len(df_list) == 1:
        return df_list[0]

    # Start with first two dataframes
    df1 = df_list[0]
    df2 = df_list[1]
    w1 = weight_values.get(df_names[0], 1.0)
    w2 = weight_values.get(df_names[1], 1.0)

    # Normalize weights
    total_weight = w1 + w2
    w1_norm = w1 / total_weight
    w2_norm = w2 / total_weight

    # Use the weighted merge function with configurable limit
    result = weighted_merge_spark(df1, df2, w1_norm, w2_norm, limit)

    # If there are more dataframes, merge them iteratively
    for i in range(2, len(df_list)):
        additional_df = df_list[i]
        additional_weight = weight_values.get(df_names[i], 1.0)

        # For subsequent merges, treat current result as one unit
        # and new dataframe as another
        current_weight = 1.0
        total_weight = current_weight + additional_weight
        w1_norm = current_weight / total_weight
        w2_norm = additional_weight / total_weight

        result = weighted_merge_spark(result, additional_df, w1_norm, w2_norm, limit)

    return result


def weighted_merge_spark(df1, df2, w1, w2, limit):
    # Calculate quotas
    q1 = round(limit * w1)
    q2 = limit - q1

    # Assign row numbers based on rank
    w_rank1 = Window.orderBy("rank")
    w_rank2 = Window.orderBy("rank")

    df1_ranked = df1.withColumn("rn", row_number().over(w_rank1))
    df2_ranked = df2.withColumn("rn", row_number().over(w_rank2))

    # Select top rows based on quota
    df1_top = df1_ranked.filter(col("rn") <= q1).drop("rn")
    df2_top = df2_ranked.filter(col("rn") <= q2).drop("rn")

    # Combine, drop duplicates, sort by rank
    merged = (
        df1_top.union(df2_top)
        .dropDuplicates(["source", "target"])
        .orderBy("rank")
        .limit(limit)
    )

    return merged
