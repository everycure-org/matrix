"""This module annotates prediction data and runs statistical analyses on indicted/nonindicted pairs."""
# NOTE: This file was partially generated using AI assistance.

import logging
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Set, Tuple

import pandas as pd
import requests
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def normalize_curies(df: pd.DataFrame, cache: Dict[str, Dict[str, str]], params: Dict) -> Tuple[Dict, List[str]]:
    """Normalize CURIEs and update cache.

    Args:
        df: Input DataFrame containing 'source' and 'target' columns.
        cache: Dictionary caching normalized CURIEs.
        params: Dictionary containing parameters for normalization.

    Returns:
        A tuple containing the updated cache and a list of failed CURIEs.
    """
    MAX_RETRIES = params["MAX_RETRIES"]
    BATCH_SIZE = params["BATCH_SIZE"]
    NORMALIZATION_URL = params["NORMALIZATION_URL"]

    # Combine 'source' and 'target' columns processing
    df[["source", "target"]] = df[["source", "target"]].apply(
        lambda col: col.astype(str).str.strip().str.replace(" ", "")
    )

    # Extract unique CURIEs from both 'source' and 'target'
    curies_series = (
        df[["source", "target"]].apply(lambda col: col.str.split("[,;]")).stack().explode().dropna().unique()
    )
    all_curies = {curie for curie in curies_series if ":" in curie and len(curie.split(":")) == 2}
    logger.info(f"Total unique CURIEs collected: {len(all_curies)}")

    unknown_curies = [curie for curie in all_curies if curie not in cache]
    logger.info(f"CURIEs to be normalized (not in cache): {len(unknown_curies)}")

    batches = [unknown_curies[i : i + BATCH_SIZE] for i in range(0, len(unknown_curies), BATCH_SIZE)]
    logger.info(f"Total batches to process: {len(batches)}")

    failed_ids: Set[str] = set()
    session = requests.Session()
    for idx, batch in enumerate(batches, start=1):
        logger.info(f"Processing batch {idx}/{len(batches)} with {len(batch)} CURIEs...")
        payload = {
            "curies": batch,
            "conflate": False,
            "description": False,
            "drug_chemical_conflate": False,
        }

        retries = 0
        while True:
            try:
                response = session.post(NORMALIZATION_URL, json=payload)
                if response.status_code == 200:
                    response_data = response.json()
                    if not isinstance(response_data, dict):
                        raise Exception("Unexpected response data type")

                    # Collect updates in a dictionary
                    updates = {}
                    for key, value in response_data.items():
                        if isinstance(value, dict):
                            identifier = value.get("id", {}).get("identifier")
                            label = value.get("id", {}).get("label")
                            if identifier and label:
                                updates[key] = {"identifier": identifier, "label": label}
                            else:
                                failed_ids.add(key)
                        else:
                            failed_ids.add(key)
                    # Update cache in bulk
                    cache.update(updates)
                    break
                if retries < MAX_RETRIES:
                    retries += 1
                    sleep(2**retries)
                else:
                    failed_ids.update(payload["curies"])
                    break
            except Exception as e:
                logger.warning(f"Exception during normalization: {e}")
                if retries < MAX_RETRIES:
                    retries += 1
                    sleep(2**retries)
                else:
                    failed_ids.update(payload["curies"])
                    break
    # Convert failed_ids set to list before returning
    return cache, list(failed_ids)


def apply_normalization_to_df(
    df: pd.DataFrame,
    cache: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Apply normalization to the DataFrame using cached CURIE normalizations.

    Args:
        df: The input DataFrame containing 'source' and 'target' columns.
        cache: Dictionary containing cached normalized CURIEs.

    Returns:
        The DataFrame with added normalized columns.
    """
    logger.info("Applying normalization to dataframe...")
    # Ensure 'source' and 'target' are strings
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)

    def get_normalization(row):
        source_cache = cache.get(row["source"], {})
        target_cache = cache.get(row["target"], {})
        return pd.Series(
            {
                "normalized_source_id": source_cache.get("identifier", ""),
                "normalized_source_label": source_cache.get("label", ""),
                "normalized_target_id": target_cache.get("identifier", ""),
                "normalized_target_label": target_cache.get("label", ""),
            }
        )

    df[["normalized_source_id", "normalized_source_label", "normalized_target_id", "normalized_target_label"]] = (
        df.apply(get_normalization, axis=1)
    )
    return df


def extract_officially_indicated_pairs(
    normalized_df: pd.DataFrame, tsv_df: pd.DataFrame, approval_cache: Dict[str, Any], params: Dict
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Extract officially indicated pairs by leveraging a cache and a TSV file.

    Args:
        normalized_df: DataFrame containing normalized data with 'normalized_source_id' column.
        tsv_df: DataFrame from TSV file containing 'drug_ID' and 'disease_ID' columns.
        approval_cache: Dictionary caching approval data.
        params: Dictionary containing parameters for extraction.

    Returns:
        A tuple containing the updated approval cache and a dictionary of officially indicated pairs.
    """
    MAX_RETRIES = params["MAX_RETRIES"]
    BATCH_SIZE = params["BATCH_SIZE"]
    MAX_WORKERS = params["MAX_WORKERS"]
    API_URL = params["API_URL"]

    # Clean 'normalized_source_id' column
    normalized_df["normalized_source_id"] = normalized_df["normalized_source_id"].astype(str)
    normalized_df = normalized_df[
        normalized_df["normalized_source_id"].notnull()
        & (normalized_df["normalized_source_id"] != "")
        & (normalized_df["normalized_source_id"].str.lower() != "nan")
    ]

    source_ids = normalized_df["normalized_source_id"].unique().tolist()
    source_ids_to_fetch = [sid for sid in source_ids if sid not in approval_cache]
    logger.info(f"Number of source_ids to fetch: {len(source_ids_to_fetch)}")

    if not source_ids_to_fetch:
        logger.info("No new source_ids to fetch. Proceeding with existing approval cache.")
    else:
        # Filter out invalid source id values
        def is_valid_sid(sid):
            return isinstance(sid, str) and sid and sid.lower() != "nan" and sid.strip() != ""

        source_ids_to_fetch = [sid for sid in source_ids_to_fetch if is_valid_sid(sid)]

        if not source_ids_to_fetch:
            logger.info("No supported source_ids to fetch after filtering.")
        else:
            batches = [source_ids_to_fetch[i : i + BATCH_SIZE] for i in range(0, len(source_ids_to_fetch), BATCH_SIZE)]

            def fetch_approval_batch(source_ids_batch: List[str]) -> Dict[str, Any]:
                payload = {"ids": source_ids_batch}
                headers = {"Content-Type": "application/json"}
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        logger.info(
                            f"Attempt {attempt}: Sending batch request for {len(source_ids_batch)} source_ids..."
                        )
                        response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
                        if response.status_code == 200:
                            logger.info(f"Batch request successful for batch starting with {source_ids_batch[0]}")
                            return response.json()
                        else:
                            logger.warning(f"Attempt {attempt}: Failed with status code {response.status_code}")
                    except requests.RequestException as e:
                        logger.warning(f"Attempt {attempt}: Exception during batch request: {e}")
                    sleep_time = min(2**attempt, 60)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    sleep(sleep_time)
                logger.error(
                    f"Failed to fetch approval data after {MAX_RETRIES} attempts for batch starting with {source_ids_batch[0]}"
                )
                return {}

            fetched_batches = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_batch = {executor.submit(fetch_approval_batch, batch): batch for batch in batches}
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        data = future.result()
                        if data:
                            fetched_batches.append(data)
                    except Exception as e:
                        logger.error(f"Exception during fetching batch starting with {batch[0]}: {e}")
                        continue

            for fetched_data in fetched_batches:
                approval_cache.update(fetched_data)

    # Process tsv_df to create tsv_drug_disease mapping
    tsv_drug_disease = tsv_df.groupby("drug_ID")["disease_ID"].apply(set).to_dict()

    # Build officially_indicated_pairs
    officially_indicated_pairs = {}
    for source_id, data in approval_cache.items():
        if not data or not isinstance(data, dict):
            continue
        chembl_data = data.get("chembl", {})
        drug_indications = chembl_data.get("drug_indications", [])

        mondo_ids = [
            efo.get("id") for indication in drug_indications for efo in indication.get("efo", []) if efo.get("id")
        ]
        if mondo_ids:
            officially_indicated_pairs[source_id] = list(set(mondo_ids))

    # Merge tsv_drug_disease into officially_indicated_pairs
    for drug_id, disease_ids in tsv_drug_disease.items():
        if drug_id in officially_indicated_pairs:
            existing_disease_ids = set(officially_indicated_pairs[drug_id])
            merged_disease_ids = existing_disease_ids.union(disease_ids)
            officially_indicated_pairs[drug_id] = sorted(merged_disease_ids)
        else:
            officially_indicated_pairs[drug_id] = sorted(disease_ids)

    return approval_cache, officially_indicated_pairs


def annotate_indicated_pairs(df: pd.DataFrame, officially_indicated_pairs: Dict[str, List[str]]) -> pd.DataFrame:
    """Annotate pairs in the DataFrame if they are officially indicated pairs.

    Args:
        df: DataFrame containing 'normalized_source_id' and 'normalized_target_id' columns.
        officially_indicated_pairs: Dictionary mapping drugs to their indicated diseases.

    Returns:
        DataFrame with an additional 'indicated_pair' boolean column.
    """
    # Create a set of indicated pairs for fast lookup
    indicated_pairs_set = {
        (drug_id, disease_id)
        for drug_id, disease_ids in officially_indicated_pairs.items()
        for disease_id in disease_ids
    }

    # Create a Series of tuples from DataFrame
    df["pair_tuple"] = list(zip(df["normalized_source_id"], df["normalized_target_id"]))

    # Check if each pair is in the set
    df["indicated_pair"] = df["pair_tuple"].isin(indicated_pairs_set)

    # Drop the temporary 'pair_tuple' column
    df = df.drop(columns=["pair_tuple"])

    return df


def filter_approved_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter approved rows from the DataFrame.

    Args:
        df: DataFrame containing an 'approved' column.

    Returns:
        DataFrame filtered to only approved rows.
    """
    if df["approved"].dtype != bool:
        df["approved"] = df["approved"].astype(str).str.strip().str.lower()
        true_values = {"true", "1", "yes", "y", "t"}
        df["approved"] = df["approved"].isin(true_values)
    approved_df = df[df["approved"] is True]
    return approved_df


def compute_statistics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute statistical tests between 'indicated' and 'non-indicated' groups.

    Args:
        df: DataFrame containing 'indicated_pair' and score columns.
        params: Dictionary containing parameters for statistics.

    Returns:
        A list of dictionaries containing statistical test results.
    """

    indicator_column = "indicated_pair"
    score_column = "score"

    # Ensure the indicator column has exactly two unique values
    unique_values = df[indicator_column].dropna().unique()
    if len(unique_values) != 2:
        raise ValueError(f"Indicator column '{indicator_column}' does not have exactly two unique values.")

    # Ensure the score column is numeric
    if not pd.api.types.is_numeric_dtype(df[score_column]):
        df = df.dropna(subset=[score_column])
        df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
        df = df.dropna(subset=[score_column])

    # Assign group labels
    df["group"] = np.where(df[indicator_column], "indicated", "non-indicated")

    # Separate scores into two groups
    non_indicated_scores = df[df["group"] == "non-indicated"][score_column]
    indicated_scores = df[df["group"] == "indicated"][score_column]

    # Check sample sizes
    n_non_indicated = len(non_indicated_scores)
    n_indicated = len(indicated_scores)

    # Perform Welch's t-test
    t_stat, t_p_value = stats.ttest_ind(non_indicated_scores, indicated_scores, equal_var=False)

    # Perform Kolmogorov-Smirnov test
    ks_stat, ks_p_value = stats.ks_2samp(non_indicated_scores, indicated_scores)

    # Adjust p-values using Bonferroni correction
    p_values = [t_p_value, ks_p_value]
    adjusted_p_values = multipletests(p_values, method="bonferroni")[1]
    adjusted_p_value_t_test = adjusted_p_values[0]
    adjusted_p_value_ks = adjusted_p_values[1]

    # Convert statistics to native Python types
    stats_list = [
        {
            "test": "t_test",
            "statistic": float(t_stat),
            "p_value": float(t_p_value),
            "adjusted_p_value": float(adjusted_p_value_t_test),
        },
        {
            "test": "ks_test",
            "statistic": float(ks_stat),
            "p_value": float(ks_p_value),
            "adjusted_p_value": float(adjusted_p_value_ks),
        },
        {"group": "non-indicated", "sample_size": int(n_non_indicated)},
        {"group": "indicated", "sample_size": int(n_indicated)},
    ]
    return stats_list
