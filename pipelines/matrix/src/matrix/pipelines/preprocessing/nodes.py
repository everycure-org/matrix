import asyncio
import json
import logging
from typing import Iterable, List, Tuple

import aiohttp
import pandas as pd
import requests
from matrix.utils.pa_utils import Column, DataFrameSchema, check_output
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def coalesce(s: pd.Series, *series: List[pd.Series]):
    """Coalesce the column information like a SQL coalesce."""
    for other in series:
        s = s.mask(pd.isnull, other)
    return s


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
async def resolve_name_async(name: str, cols_to_get: Iterable[str], url: str, session: aiohttp.ClientSession) -> dict:
    """Asynchronously retrieve the normalized identifier through the normalizer.

    Args:
        name: name of the node to be resolved
        cols_to_get: attribute to get from API
        url: URL template for the API
        session: aiohttp client session
    Returns:
        Name and corresponding curie
    """
    if not name or pd.isna(name):
        return {}

    try:
        async with session.get(url.format(name=name)) as response:
            result = await response.json()

        if not result:
            return {}

        element = result[0]
        ret = {col: element.get(col) for col in cols_to_get}
        return ret

    except Exception as e:
        logger.error(f"Error resolving name {name}: {str(e)}")
        raise e


async def resolve_names_batch(names: pd.Series, cols_to_get: Iterable[str], url: str) -> List[dict]:
    """Resolve a batch of names asynchronously.

    Args:
        names: Series of names to resolve
        cols_to_get: attributes to get from API
        url: URL template for the API
    Returns:
        List of resolved name data
    """
    async with aiohttp.ClientSession() as session:
        tasks = [resolve_name_async(name, cols_to_get, url, session) for name in names]
        return await asyncio.gather(*tasks)


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
        unique=["drug_curie", "disease_curie"],
    )
)
def add_source_and_target_to_clinical_trails(df: pd.DataFrame, resolver_url: str) -> pd.DataFrame:
    """Resolve names to curies for source and target columns in clinical trials data.

    Args:
        df: Clinical trial dataset
    """
    # Normalize names asynchronously
    drug_data = asyncio.run(resolve_names_batch(df["drug_name"], ["curie"], resolver_url))
    disease_data = asyncio.run(resolve_names_batch(df["disease_name"], ["curie"], resolver_url))

    # Concat dfs
    drug_df = pd.DataFrame(drug_data).rename(columns={"curie": "drug_curie"})
    disease_df = pd.DataFrame(disease_data).rename(columns={"curie": "disease_curie"})
    df = pd.concat([df, drug_df, disease_df], axis=1)

    return df


@check_output(
    schema=DataFrameSchema(
        columns={
            "curie": Column(str, nullable=False),
            "name": Column(str, nullable=False),
        },
        unique=["curie"],
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
def clean_clinical_trial_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    nodes = pd.concat([drugs, diseases], ignore_index=True).drop_duplicates(subset="curie")
    return {"nodes": nodes, "edges": edges}
