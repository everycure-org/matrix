import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _expand_search_term(term: str) -> set[str]:
    normalized_term = term.strip()
    if normalized_term == "":
        return set()

    search_terms = {normalized_term}
    words = normalized_term.split()
    if len(words) == 2:
        search_terms.add(f"{words[0]}, {words[1]}")
        search_terms.add(f"{words[1]}, {words[0]}")

    return search_terms


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


def filter_dataframe_by_columns(df: pd.DataFrame, filter_columns: dict[str, str]) -> pd.DataFrame:
    """
    Filter a DataFrame by requiring each specified column to equal its expected value.

    Columns listed in `filter_columns` that are not present in `df` are skipped with
    a warning rather than raising an error, so callers can pass a superset of possible
    filter keys without needing to guard against missing columns.

    Args:
        df: The dataframe to filter.
        filter_columns: A mapping of column name → expected value. Rows are kept only
            when every present column matches its expected value.

    Returns:
        A new dataframe containing only the rows that satisfy all filter conditions.
    """

    for col, expected_value in filter_columns.items():
        if col in df.columns:
            df = df[df[col] == expected_value]
        else:
            logger.error(f"Filter column {col} not found in dataframe")
    return df


def create_search_term_from_curated_drug_list(
    curated_drug_list: pd.DataFrame, curated_drug_list_columns_to_use_for_matching: list[str]
) -> pd.DataFrame:
    """
    Create a dataframe with search terms for each drug based on specified columns.
    Args:
    curated_drug_list: The dataframe containing curated drugs
    curated_drug_list_columns_to_use_for_matching: List of column names to use for generating
        search terms (e.g., "name", "synonyms", "brand_names")

    Returns:
        A dataframe with columns "name", "id", "search_terms", and "available_in_combo_with", where "search_terms" is a set of terms generated from the specified columns for each drug
    """
    work_items = []
    for _, row in curated_drug_list.iterrows():
        search_terms = set()
        for col in curated_drug_list_columns_to_use_for_matching:
            value = row[col]
            if isinstance(value, (list, np.ndarray)):
                for item in value:
                    if isinstance(item, str):
                        search_terms.update(_expand_search_term(item))
            elif isinstance(value, str):
                search_terms.update(_expand_search_term(value))

        work_items.append(
            {
                "name": row["name"],
                "id": row["id"],
                "search_terms": search_terms,
                "available_in_combo_with": row["available_in_combo_with"],
            }
        )
    return pd.DataFrame(work_items)
