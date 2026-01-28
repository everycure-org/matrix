import logging

import pandas as pd

logger = logging.getLogger(__name__)


def apply_patch(df: pd.DataFrame, patch_df: pd.DataFrame, patch_columns: list[str], merge_on: str) -> pd.DataFrame:
    merged_df = pd.merge(df, patch_df, on=merge_on, how="left", suffixes=("", "_patch"))

    for col in patch_columns:
        patch_col = f"{col}_patch"
        if patch_col in merged_df.columns:
            mask = merged_df[patch_col].notna()
            merged_df.loc[mask, col] = merged_df.loc[mask, patch_col]
            merged_df = merged_df.drop(columns=[patch_col])
        else:
            raise ValueError(f"Patch column {col} not found in df")

    return merged_df


def _log_merge_statistics(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    primary_name: str,
    secondary_name: str,
    id_column: str = "id",
    primary_only_action: str = "will be kept with nulls",
    secondary_only_action: str = "will be dropped",
) -> None:
    """
    Log statistics about merging two dataframes, showing differences in IDs.

    Args:
        primary_df: The primary dataframe
        secondary_df: The secondary dataframe
        primary_name: Name for the primary dataframe in logs
        secondary_name: Name for the secondary dataframe in logs
        id_column: Column name to use for ID comparison
        primary_only_action: Description of what happens to primary-only IDs
        secondary_only_action: Description of what happens to secondary-only IDs
    """
    primary_ids = set(primary_df[id_column])
    secondary_ids = set(secondary_df[id_column])

    primary_only_ids = primary_ids - secondary_ids
    secondary_only_ids = secondary_ids - primary_ids
    matching_ids = primary_ids & secondary_ids

    if primary_only_ids or secondary_only_ids:
        logger.warning(f"Merge statistics for {primary_name} and {secondary_name}:")
        logger.warning(f"  - {primary_name} total rows: {len(primary_df)}")
        logger.warning(f"  - {secondary_name} total rows: {len(secondary_df)}")
        logger.warning(f"  - Matching IDs: {len(matching_ids)}")
        logger.warning(f"  - {primary_name} only ({primary_only_action}): {len(primary_only_ids)}")
        logger.warning(f"  - {secondary_name} only ({secondary_only_action}): {len(secondary_only_ids)}")

        if primary_only_ids:
            sample_primary_only = list(primary_only_ids)[:5]  # Show first 5
            logger.warning(f"  - Sample {primary_name}-only IDs: {sample_primary_only}")

        if secondary_only_ids:
            sample_secondary_only = list(secondary_only_ids)[:5]  # Show first 5
            logger.warning(f"  - Sample {secondary_name}-only IDs: {sample_secondary_only}")
