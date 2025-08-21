import logging
import math
import random

import pandas as pd
import pyspark.sql as ps
from pandera import Column, DataFrameSchema, check_output
from pyspark.sql.functions import col

logger = logging.getLogger(__name__)


def prefetch_top_quota(
    weights: dict[str, dict],
    config: dict[str, any],
    **dataframes: ps.DataFrame,
) -> list[ps.DataFrame]:
    """
    Return the top rows from each input DataFrame according to its quota with a 20% buffer.

    - Quota is computed as (limit * weight) per dataset.
    - We double the quota as a buffer, in case there is significant similarity between datasets.

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

    # Apply 2x buffer to handle many duplicates when high commonality between datasets.
    dataset_names = list(dataframes.keys())
    buffered_quota = [
        math.ceil((limit * weights[name]["weight"]) * 2) for name in dataset_names
    ]

    trimmed_dataframes: list[ps.DataFrame] = []
    for name, buffered_quota in zip(dataset_names, buffered_quota):
        top_plus_buffer = dataframes[name].filter(col("rank") <= buffered_quota)
        trimmed_dataframes.append(top_plus_buffer)

    return trimmed_dataframes


@check_output(
    DataFrameSchema(
        columns={
            "source": Column(
                str, nullable=False, description="Source entity identifier"
            ),
            "target": Column(
                str, nullable=False, description="Target entity identifier"
            ),
            "from_input_datasets": Column(
                str,
                nullable=False,
                description="Name of the input dataset(s) that provided the row",
            ),
            "rank": Column(
                int, nullable=False, description="Ranking position starting from 1"
            ),
        },
        unique=["source", "target", "rank"],
    ),
)
def weighted_interleave_dataframes(  # noqa: PLR0912
    weights: dict[str, dict],
    config: dict[str, any],
    rng: random.Random | None = None,
    **trimmed_dataframes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform weighted interleaving of trimmed DataFrames.

    For each row position, randomly select a dataset according to weights,
    then take the top-ranked item from that dataset.
    Avoid duplicates by tracking seen (source, target) pairs.

    Args:
        weights: Mapping of dataset name to a dict containing a 'weight' float
        config: Mapping containing 'limit' int
        rng: Optional random instance. If provided, used for deterministic tests.
        **trimmed_dataframes: Named trimmed pandas DataFrames keyed by dataset name

    Returns:
        pandas.DataFrame: Interleaved DataFrame with sequential ranks
    """
    if not trimmed_dataframes:
        raise ValueError("At least one DataFrame must be provided")

    if "limit" not in config:
        raise ValueError("Missing limit in config")
    limit = config["limit"]

    # Extract and normalize weights to sum to 1
    total_weight = sum(weights[name]["weight"] for name in trimmed_dataframes.keys())
    normalized_weights = {
        name: weights[name]["weight"] / total_weight
        for name in trimmed_dataframes.keys()
    }

    # Make copies and initialize tracking
    pandas_dfs = {name: df.copy() for name, df in trimmed_dataframes.items()}
    result_rows = []
    seen_pairs = set()
    dataset_names = list(trimmed_dataframes.keys())

    # Use provided RNG for unit
    _rng = rng or random

    # Continue until we reach the limit or run out of unique rows
    attempts = 0
    while len(result_rows) < limit and attempts < limit * 10:
        attempts += 1

        # Select dataset by weight
        selected_dataset = _rng.choices(
            dataset_names,
            weights=[normalized_weights[name] for name in dataset_names],
        )[0]

        df = pandas_dfs[selected_dataset]
        # If the dataset is empty, skip it. This may happen if we exhaust all rows in the input list.
        if df.empty:
            continue

        # Find the first available pair that hasn't been seen
        available_row = None
        rows_to_remove = []

        for idx, row in df.iterrows():
            pair_key = (row["source"], row["target"])
            if pair_key in seen_pairs:
                # Update the existing result row to include this dataset
                for result_row in result_rows:
                    if (
                        result_row["source"] == row["source"]
                        and result_row["target"] == row["target"]
                    ):
                        result_row["from_input_datasets"] += f",{selected_dataset}"
                        break
                rows_to_remove.append(idx)
            else:
                # Found the first non-duplicate row
                available_row = row
                break

        # If no available row found, skip this dataset
        if available_row is None:
            # We still need to remove duplicate rows
            if rows_to_remove:
                pandas_dfs[selected_dataset] = df.drop(rows_to_remove)
            continue

        # Create new row with the new column
        new_row = {
            "source": available_row["source"],
            "target": available_row["target"],
            "from_input_datasets": selected_dataset,
        }
        # Add to results
        result_rows.append(new_row)

        # Track seen pairs
        pair_key = (available_row["source"], available_row["target"])
        seen_pairs.add(pair_key)

        # Remove the selected row and any duplicates
        pandas_dfs[selected_dataset] = df.drop(rows_to_remove + [available_row.name])

    if not result_rows:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=["source", "target", "from_input_datasets", "rank"])

    # Create result with sequential ranks
    result_df = pd.DataFrame(result_rows).reset_index(
        drop=True
    )  # reset original Series index
    if len(result_df) < limit:
        logger.warning(
            f"Requested limit {limit} but only {len(result_df)} unique rows exist."
        )
    result_df["rank"] = range(1, len(result_df) + 1)

    return result_df
