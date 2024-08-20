"""Nodes for the preprocessing pipeline."""
import requests

import pandas as pd
import numpy as np

from typing import Callable, List, Optional
from functools import partial

from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key


def resolve(name: str, endpoint: str) -> str:
    """Function to retrieve curie through the synonymizer.

    Args:
        name: name of the node
        endpoint: endpoint of the synonymizer
    Returns:
        Corresponding curie
    """
    result = requests.get(f"{endpoint}/synonymize", json={"names": [name]})

    element = result.json().get(name)
    if element:
        return element.get("preferred_curie", None)

    return None


def normalize(curie: str, endpoint: str):
    """Function to retrieve the normalized identifier through the synonymizer.

    Args:
        curie: curie of the node
        endpoint: endpoint of the synonymizer
    Returns:
        Corresponding curie
    """
    if not curie or pd.isna(curie):
        return None

    result = requests.get(f"{endpoint}/normalize", json={"names": [curie]})

    element = result.json().get(curie)
    if element:
        return element.get("id", {}).get("identifier")

    return None


def coalesce(s: pd.Series, *series: List[pd.Series]):
    """Coalesce the column information like a SQL coalesce."""
    for other in series:
        s = s.mask(pd.isnull, other)
    return s


@has_schema(
    schema={
        "ID": "numeric",
        "name": "object",
        "curie": "object",
    },
    allow_subset=True,
)
@primary_key(primary_key=["ID"])
def enrich_df(
    df: pd.DataFrame,
    endpoint: str,
    func: Callable,
    input_cols: str,
    target_col: str,
    coalesce_col: Optional[str] = None,
) -> pd.DataFrame:
    """Function to resolve nodes of the nodes input dataset.

    Args:
        df: nodes dataframe
        endpoint: endpoint of the synonymizer
        func: func to call
        input_cols: input cols, cols are coalesced to obtain single column
        target_col: target col
        coalesce_col: col to coalesce with if None
    Returns:
        dataframe enriched with Curie column
    """
    # Replace empty strings with nan for processing by pandas, we revert back at end
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Coalesce input cols
    col = coalesce(*[df[col] for col in input_cols])

    # Apply enrich function and replace nans by empty space
    df[target_col] = col.apply(partial(func, endpoint=endpoint))

    # If set, coalesce final result with coalesce col
    if coalesce_col:
        df[target_col] = coalesce(df[coalesce_col], df[target_col])

    # Ensure to fill nans with empty strings to avoid nans in nodebook
    return df.fillna("")


def create_int_nodes(int_nodes: pd.DataFrame) -> pd.DataFrame:
    """Function to create a intermediate nodes dataset by filtering and renaming columns."""
    # Replace empty strings with nan
    int_nodes = int_nodes.replace(r"^\s*$", np.nan, regex=True)

    return (
        int_nodes[int_nodes["normalized_curie"].notna()]
        .drop(columns="curie")
        .rename(columns={"normalized_curie": "curie"})
    )


def create_int_edges(prm_nodes: pd.DataFrame, int_edges: pd.DataFrame) -> pd.DataFrame:
    """Function to create a intermediate edges dataset by combining primary nodes with intermediate edges."""
    index = prm_nodes[["ID", "curie"]]

    res = (
        int_edges.merge(
            index.rename(columns={"curie": "SourceId"}),
            left_on="Source",
            right_on="ID",
            how="left",
        )
        .drop(columns="ID")
        .merge(
            index.rename(columns={"curie": "TargetId"}),
            left_on="Target",
            right_on="ID",
            how="left",
        )
        .drop(columns="ID")
    )

    res["Included"] = res.apply(
        lambda row: not (pd.isna(row["SourceId"]) or pd.isna(row["TargetId"])), axis=1
    )

    return res.fillna("")


@has_schema(
    schema={
        "category": "object",
        "id": "object",
        "name": "object",
        "description": "object",
    },
    allow_subset=True,
)
@primary_key(primary_key=["id"])
def create_prm_nodes(prm_nodes: pd.DataFrame) -> pd.DataFrame:
    """Function to create a primary nodes that contains only new nodes introduced by the source."""
    # `new_id` signals that the node should be added to the KG as a new id
    # we drop the original ID from the spreadsheat, and leverage the new_id as the final id
    # in the dataframe. We only retain nodes where the new_id is set
    res = (
        prm_nodes[prm_nodes["new_id"].notna()]
        .drop(columns="ID")
        .rename(columns={"new_id": "id"})
    )

    res["category"] = "biolink:" + prm_nodes["entity label"]

    return res

@has_schema(
    schema={
        "subject": "object",
        "predicate": "object",
        "object": "object",
        "knowledge_source": "object",
    },
    allow_subset=True,
)

def create_prm_edges(int_edges: pd.DataFrame) -> pd.DataFrame:
    """Function to create a primary edges dataset by filtering and renaming columns."""
    # Replace empty strings with nan
    res = (
        int_edges.replace(r"^\s*$", np.nan, regex=True)
        .rename(
            columns={"SourceId": "subject", "TargetId": "object", "Label": "predicate"}
        )
        .dropna(subset=["subject", "object"])
    )

    res["predicate"] = "biolink:" + res["predicate"]
    res["knowledge_source"] = "ec:medical"

    return res

@has_schema(
    schema={
        "clinical_trial_id": "object",
        "reason_for_rejection": "object",
        "drug_name": "object",
        "disease_name": "object",
        "significantly_better": "object",
        "non_significantly_better": "object",
        "non_significantly_worse": "object",
        "significantly_worse": "object"
    },
    allow_subset=True,
    df="df"
)
def map_name_to_curie(
    df: pd.DataFrame,
    endpoint: str,
) -> pd.DataFrame:
    """Function to map drug name or disease name in raw clinical trail dataset to curie using the synonymizer.
       Original Clinical trial data should be a EXCEL format containg 8 columns:
       clinical_trial_id, reason_for_rejection, drug_name, disease_name, significantly_better, non_significantly_better, non_significantly_worse, significantly_worse

    Args:
        df: raw clinical trial dataset from medical team
        endpoint: endpoint of the synonymizer
    Returns:
        dataframe with two additional columns: "Mapped Drug Curie" and "Mapped Drug Disease"
    """
    
    # Map the drug name to the corresponding curie ids
    df["drug_kg_curie"] = df["drug_name"].apply(lambda x: resolve(x, endpoint=endpoint))
    # Map the disease name to the corresponding curie ids
    df["disease_kg_curie"] = df["disease_name"].apply(lambda x: resolve(x, endpoint=endpoint))
    
    return df


@has_schema(
    schema={
        "clinical_trial_id": "object",
        "reason_for_rejection": "object",
        "drug_name": "object",
        "disease_name": "object",
        "significantly_better": "object",
        "non_significantly_better": "object",
        "non_significantly_worse": "object",
        "significantly_worse": "object"
        "drug_kg_curie": "object",
        "disease_kg_curie": "object"
    },
    allow_subset=True,
    df="df"
)
def clean_clinical_trial_data(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Function to clean the mapped clinical trial dataset for use in time-split evaluation metrics.
       Clinical trial data should be a EXCEL format containg 10 columns:
       clinical_trial_id, reason_for_rejection, drug_name, disease_name, significantly_better, non_significantly_better, non_significantly_worse, significantly_worse, drug_kg_curie, disease_kg_curie
       
       2. We filter out rows with the following conditions:
            - Missing drug_kg_curie
            - Missing disease_kg_curie
            - with reason for rejection
            - missing values in either significantly_better, non_significantly_better, non_significantly_worse, or significantly_worse columns

       3. Only keep the following columns:
            - drug_name
            - disease_name
            - significantly_better
            - non_significantly_better
            - non_significantly_worse
            - significantly_worse
            - drug_kg_curie
            - disease_kg_curie
    Args:
        df: raw clinical trial dataset added with mapped drug and disease curies
    Returns:
        Cleaned clinical trial data.
    """

    # remove rows with reason for rejection
    df = df[
        df["reason_for_rejection"].map(lambda x: type(x) != str)
    ].reset_index(drop=True)

    # remove rows with missing drug_kg_curie or disease_kg_curie
    row_has_missing = (
        df["drug_kg_curie"].isna()
        | df["disease_kg_curie"].isna()
    )
    df = df[~row_has_missing].reset_index(drop=True)

    # remove rows with missing values in significantly better, non-significantly better, non-significantly worse, or significantly worse columns
    row_has_missing = (
        df["significantly_better"].isna()
        | df["non_significantly_better"].isna()
        | df["non_significantly_worse"].isna()
        | df["significantly_worse"].isna()
    )
    df = df[~row_has_missing].reset_index(drop=True)

    # drop columns: clinical_trial_id, reason_for_rejection
    df = df.drop(
        columns=["clinical_trial_id", "reason_for_rejection"]
    )

    return df