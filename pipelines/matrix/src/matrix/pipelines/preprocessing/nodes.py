import logging
import random
import time
from typing import Collection, Dict, List, Sequence

import pandas as pd
import pyspark.sql.functions as f
import requests
from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output
from tenacity import Retrying, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

import asyncio
from typing import Dict, List

import aiohttp
import nest_asyncio
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_json, struct, to_json
from pyspark.sql.types import ArrayType, StringType, StructField, StructType
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

nest_asyncio.apply()


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def process_batch(batch: List[str], session: aiohttp.ClientSession, pbar: tqdm) -> Dict:
    """Process a single batch of IDs with retry logic"""
    url = "https://nodenorm.test.transltr.io/1.5/get_normalized_nodes"
    results = {}

    # Filter out any empty or None values
    batch = [x for x in batch if x and isinstance(x, str)]

    payload = {"curies": batch, "conflate": True, "description": False, "drug_chemical_conflate": False}

    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                for curie, node_data in result.items():
                    if node_data:
                        results[curie] = {
                            "id": node_data["id"]["identifier"] if node_data.get("id") else None,
                            "label": node_data["id"].get("label", None) if node_data.get("id") else None,
                            "all_categories": node_data.get("type", None),
                            "original_id": curie,
                        }
                    else:
                        results[curie] = {"id": None, "label": None, "all_categories": None, "original_id": curie}
            elif response.status == 502:
                print(f"Server overloaded (502). Retrying batch after delay...")
                await asyncio.sleep(10)  # Longer delay for server errors
                raise aiohttp.ClientError("Server overloaded")
            elif response.status == 422:
                print(f"Invalid data in batch (422). Skipping problematic IDs...")
                print(f"Problematic batch: {batch}")
                return {}
            else:
                print(f"Error status {response.status}")

    except Exception as e:
        print(f"Error processing batch: {e}")
        raise

    pbar.update(1)
    return results


async def resolve_ids_batch_async(curies: List[str], batch_size: int = 250, max_concurrent: int = 10) -> Dict:
    """
    Resolve IDs using concurrent async POST requests with improved error handling
    """
    all_results = {}

    # Clean input data
    curies = [str(x) for x in curies if x is not None]

    # Split into batches
    batches = [curies[i : i + batch_size] for i in range(0, len(curies), batch_size)]
    total_batches = len(batches)

    pbar = tqdm(total=total_batches, desc="Processing batches")

    connector = aiohttp.TCPConnector(
        limit=max_concurrent, force_close=False, enable_cleanup_closed=True, ttl_dns_cache=300
    )

    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(batch):
            async with semaphore:
                try:
                    return await process_batch(batch, session, pbar)
                except Exception as e:
                    print(f"Batch failed: {e}")
                    return {}

        for i in range(0, len(batches), max_concurrent):
            batch_group = batches[i : i + max_concurrent]
            tasks = [process_with_semaphore(batch) for batch in batch_group]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, dict):
                    all_results.update(result)

            await asyncio.sleep(1)

    pbar.close()
    return all_results


def normalize_identifiers(attr: pd.DataFrame, identifiers_mapping: Dict[str, str]) -> pd.DataFrame:
    """Normalize identifiers in embiology attributes using NCATS normalizer.

    Args:
        attr: embiology attributes
        identifiers_mapping: mapping of identifiers

    Returns:
        pd.DataFrame: DataFrame with normalized identifiers
    """
    # Get values list from attribute DataFrame
    ids_to_resolve = attr["value"].tolist()

    # Run async resolution
    results = asyncio.run(resolve_ids_batch_async(curies=ids_to_resolve, batch_size=10000, max_concurrent=10))

    # Convert results to pandas DataFrame
    results_df = pd.DataFrame(
        [
            {
                "curie": k,
                "id": v["id"],
                "label": v["label"],
                "all_categories": v["all_categories"],
                "original_id": v["original_id"],
            }
            for k, v in results.items()
        ]
    )

    return results_df


def prepare_normalized_identifiers(attr: pd.DataFrame, identifiers_mapping: Dict[str, str]) -> pd.DataFrame:
    """Prepare identifiers for embiology nodes & normalize using NCATS normalizer.

    Args:
        attr: embiology attributes
        identifiers_mapping: mapping of identifiers
    """
    df = attr.withColumn(
        "value",
        f.when(
            f.col("name").isin(list(identifiers_mapping.keys())),
            f.concat(
                f.coalesce(
                    f.map_from_arrays(
                        f.array(*[f.lit(x) for x in identifiers_mapping.keys()]),
                        f.array(*[f.lit(x) for x in identifiers_mapping.values()]),
                    )[f.col("name")],
                    f.lit(""),
                ),
                f.split(f.col("value"), "\\.").getItem(0),
            ),
        ).otherwise(f.col("value")),
    )
    mapping_df = normalize_identifiers(df, identifiers_mapping)

    return mapping_df


def prepare_nodes(nodes, attr, identifiers_mapping):
    """Prepare nodes for embiology nodes.

    Args:
        nodes: embiology nodes
        identifiers_mapping: mapping of identifiers
    """
    return nodes


def resolve_one_name_batch(names: Sequence[str], url: str) -> Dict[str, List[Dict]]:
    """Batch resolve a list of names to their corresponding CURIEs."""
    payload = {
        "strings": names,
        "autocomplete": True,
        "highlighting": False,
        "offset": 0,
        "limit": 1,
    }

    for attempt in Retrying(
        wait=wait_exponential(multiplier=2, min=2, max=120), stop=stop_after_attempt(5), reraise=True
    ):
        with attempt:
            response = requests.post(url, json=payload)
            logger.debug(f"Request time: {response.elapsed.total_seconds():.2f} seconds")
            response.raise_for_status()
            return response.json()


def parse_one_name_batch(
    result: Dict[str, List[Dict[str, str]]], cols_to_get: Collection[str]
) -> Dict[str, Dict[str, str | None]]:
    """Parse API response to extract resolved names and corresponding attributes."""
    resolved_data = {}

    for name, attributes in result.items():
        if attributes:
            resolved_data[name] = {col: attributes[0].get(col) for col in cols_to_get}
        else:
            resolved_data[name] = dict.fromkeys(cols_to_get)

    return resolved_data


def resolve_names(names: Sequence[str], cols_to_get: Collection[str], url: str, batch_size: int) -> Dict[str, Dict]:
    """Function to retrieve the normalized identifier through the normalizer.

    Args:
        name: name of the node to be resolved
        cols_to_get: attribute to get from API
    Returns:
        Name and corresponding curie
    """

    resolved_data = {}
    for i in range(0, len(names), batch_size):
        batch = names[i : i + batch_size]
        logger.info(f"Resolving batch {i} of {len(names)}")
        # Waiting between requests drastically improves the API performance, as opposed to hitting a 5xx code
        # and using retrying with backoff, which can render the API unresponsive for a long time (> 10 min).
        time.sleep(random.randint(5, 10))
        batch_response = resolve_one_name_batch(batch, url)
        batch_parsed = parse_one_name_batch(batch_response, cols_to_get)
        resolved_data.update(batch_parsed)
    return resolved_data


@check_output(
    schema=DataFrameSchema(
        columns={
            "normalized_curie": Column(str, nullable=False),
            "label": Column(str, nullable=False),
            "types": Column(List[str], nullable=False),
            "category": Column(str, nullable=False),
            "ID": Column(int, nullable=False),
        },
        # NOTE: We should re-enable when the medical team fixed the dataset
        # unique=["normalized_curie"],
    )
)
def process_medical_nodes(df: pd.DataFrame, resolver_url: str, batch_size: int) -> pd.DataFrame:
    """Map medical nodes with name resolver.

    Args:
        df: raw medical nodes
        resolver_url: url for name resolver

    Returns:
        Processed medical nodes
    """
    # Normalize the name
    names = df["name"].dropna().unique().tolist()
    resolved_names = resolve_names(
        names, cols_to_get=["curie", "label", "types"], url=resolver_url, batch_size=batch_size
    )
    extra_cols = df.name.map(resolved_names)
    df = df.join(pd.json_normalize(extra_cols))

    # Coalesce id and new id to allow adding "new" nodes
    df["normalized_curie"] = df["new_id"].fillna(df["curie"])

    # Filter out nodes that are not resolved
    is_resolved = df["normalized_curie"].notna()
    df = df[is_resolved]
    if not is_resolved.all():
        logger.warning(f"{(~is_resolved).sum()} EC medical nodes have not been resolved.")

    # Flag the number of duplicate IDs
    is_unique = df["normalized_curie"].groupby(df["normalized_curie"]).transform("count") == 1
    if not is_unique.all():
        logger.warning(f"{(~is_unique).sum()} EC medical nodes are duplicated.")
    return df


@check_output(
    schema=DataFrameSchema(
        columns={
            "SourceId": Column(str, nullable=False),
            "TargetId": Column(str, nullable=False),
            "Label": Column(str, nullable=False),
        },
        # NOTE: We should re-enable when the medical team fixed the dataset
        # unique=["SourceId", "TargetId", "Label"],
    )
)
def process_medical_edges(int_nodes: pd.DataFrame, raw_edges: pd.DataFrame) -> pd.DataFrame:
    """Function to create int edges dataset.

    Function ensures edges dataset link curies in the KG.

    Args:
        int_nodes: Processed medical nodes with normalized curies
        raw_edges: Raw medical edges
    """
    df = int_nodes[["normalized_curie", "ID"]]
    # Attach source and target curies. Drop edge un the case of a missing curies.
    res = (
        raw_edges.merge(
            df.rename(columns={"normalized_curie": "SourceId"}),
            left_on="Source",
            right_on="ID",
            how="inner",
        )
        .drop(columns="ID")
        .merge(
            df.rename(columns={"normalized_curie": "TargetId"}),
            left_on="Target",
            right_on="ID",
            how="inner",
        )
        .drop(columns="ID")
    )
    return res


@check_output(
    schema=DataFrameSchema(
        columns={
            "reason_for_rejection": Column(str, nullable=True),
            "drug_name": Column(str, nullable=False),
            "disease_name": Column(str, nullable=False),
            "significantly_better": Column(float, nullable=True),
            "non_significantly_better": Column(float, nullable=True),
            "non_significantly_worse": Column(float, nullable=True),
            "significantly_worse": Column(float, nullable=True),
            "drug_curie": Column(str, nullable=True),
            "disease_curie": Column(str, nullable=True),
        },
        # NOTE: We should re-enable when the medical team fixed the dataset
        # unique=["drug_curie", "disease_curie"],
    )
)
def add_source_and_target_to_clinical_trails(df: pd.DataFrame, resolver_url: str, batch_size: int) -> pd.DataFrame:
    """Resolve names to curies for source and target columns in clinical trials data.

    Args:
        df: Clinical trial dataset
    """

    drug_names = df["drug_name"].dropna().unique().tolist()
    disease_names = df["disease_name"].dropna().unique().tolist()

    drug_mapping = resolve_names(drug_names, cols_to_get=["curie"], url=resolver_url, batch_size=batch_size)
    disease_mapping = resolve_names(disease_names, cols_to_get=["curie"], url=resolver_url, batch_size=batch_size)

    drug_mapping_df = pd.DataFrame(drug_mapping).transpose()["curie"].rename("drug_curie")
    disease_mapping_df = pd.DataFrame(disease_mapping).transpose()["curie"].rename("disease_curie")

    df = pd.merge(df, drug_mapping_df, how="left", left_on="drug_name", right_index=True)
    df = pd.merge(df, disease_mapping_df, how="left", left_on="disease_name", right_index=True)

    return df


@check_output(
    schema=DataFrameSchema(
        columns={
            "curie": Column(str, nullable=False),
            "name": Column(str, nullable=False),
        },
        # NOTE: We should re-enable when the medical team fixed the dataset
        # unique=["curie"],
    ),
    df_name="nodes",
)
@check_output(
    schema=DataFrameSchema(
        columns={
            "drug_name": Column(str, nullable=False),
            "disease_name": Column(str, nullable=False),
            "drug_curie": Column(str, nullable=False),
            "disease_curie": Column(str, nullable=False),
            "significantly_better": Column(int, nullable=False),
            "non_significantly_better": Column(int, nullable=False),
            "non_significantly_worse": Column(int, nullable=False),
            "significantly_worse": Column(int, nullable=False),
        },
        unique=["drug_curie", "disease_curie"],
    ),
    df_name="edges",
)
def clean_clinical_trial_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Clean clinical trails data.

    Function to clean the mapped clinical trial dataset for use in time-split evaluation metrics.

    Args:
        df: raw clinical trial dataset added with mapped drug and disease curies
    Returns:
        Cleaned clinical trial data.
    """
    # Columns for outcome (ordered best to worst outcome)
    outcome_columns = [
        "significantly_better",
        "non_significantly_better",
        "non_significantly_worse",
        "significantly_worse",
    ]

    # Columns for drug and disease names
    name_columns = ["drug_name", "disease_name"]

    # Remove rows with reason for rejection
    df = df[df["reason_for_rejection"].isna()]

    # Drop columns that are not needed and convert outcome columns to bool
    df = df[["drug_curie", "disease_curie", *name_columns, *outcome_columns]]

    # Drop rows with missing values in cols
    df = df.dropna().reset_index(drop=True)

    # Convert outcome column to int
    df = df.astype({col: int for col in outcome_columns})

    # Aggregate drug/disease IDs with multiple names or outcomes. Take the worst outcome.
    edges = (
        df.groupby(["drug_curie", "disease_curie"])
        .agg({"drug_name": "first", "disease_name": "first", **{col: "max" for col in outcome_columns}})
        .reset_index()
    )

    def ensure_one_true(row):
        """
        Ensure at most one outcome column is true, taking the worst outcome by convention.
        """
        for n in range(len(outcome_columns) - 1):
            if row[outcome_columns[n]] == 1:
                row[outcome_columns[n + 1 :]] = 0
        return row

    edges = edges.apply(ensure_one_true, axis=1)

    # Extract nodes
    drugs = df.rename(columns={"drug_curie": "curie", "drug_name": "name"})[["curie", "name"]]
    diseases = df.rename(columns={"disease_curie": "curie", "disease_name": "name"})[["curie", "name"]]
    nodes = pd.concat([drugs, diseases], ignore_index=True)
    return {"nodes": nodes, "edges": edges}


def report_to_gsheets(df: pd.DataFrame, sheet_df: pd.DataFrame, primary_key_col: str):
    """Report medical nodes to gsheets.

    Args:
        df: medical nodes
        sheet_df: gsheets dataframe
    """
    # TODO: in the future remove drop_duplicates function
    df = df.drop_duplicates(subset=primary_key_col, keep="first")
    return sheet_df.merge(df, on=primary_key_col, how="left").fillna("Not resolved")
