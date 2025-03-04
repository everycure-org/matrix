import json
import logging
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def resolve_name(name: str, cols_to_get: Iterable[str], url: str) -> dict:
    """Function to retrieve the normalized identifier through the normalizer.

    Args:
        name: name of the node to be resolved
        cols_to_get: attribute to get from API
    Returns:
        Name and corresponding curie
    """

    if not name or pd.isna(name):
        return {}
    result = requests.get(url.format(name=name)).json()
    if not result:
        return {}

    element = result[0]
    ret = {col: element.get(col) for col in cols_to_get}
    logger.debug(f'{{"resolver url": {url}, "name": {name}, "response extraction": {json.dumps(ret)}}}')
    return ret


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
def process_medical_nodes(df: pd.DataFrame, resolver_url: str) -> pd.DataFrame:
    """Map medical nodes with name resolver.

    Args:
        df: raw medical nodes
        resolver_url: url for name resolver

    Returns:
        Processed medical nodes
    """
    # Normalize the name
    enriched_data = df["name"].apply(resolve_name, cols_to_get=["curie", "label", "types"], url=resolver_url)

    # Extract into df
    enriched_df = pd.DataFrame(enriched_data.tolist())
    df = pd.concat([df, enriched_df], axis=1)

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
def add_source_and_target_to_clinical_trails(df: pd.DataFrame, resolver_url: str) -> pd.DataFrame:
    """Resolve names to curies for source and target columns in clinical trials data.

    Args:
        df: Clinical trial dataset
    """
    # Normalize the name
    drug_data = df["drug_name"].apply(resolve_name, cols_to_get=["curie"], url=resolver_url)
    disease_data = df["disease_name"].apply(resolve_name, cols_to_get=["curie"], url=resolver_url)

    # Concat dfs
    drug_df = pd.DataFrame(drug_data.tolist()).rename(columns={"curie": "drug_curie"})
    disease_df = pd.DataFrame(disease_data.tolist()).rename(columns={"curie": "disease_curie"})
    df = pd.concat([df, drug_df, disease_df], axis=1)

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
