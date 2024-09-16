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


def normalize(curie: str, endpoint: str, att_to_get: str = "identifier"):
    """Function to retrieve the normalized identifier through the synonymizer.

    Args:
        curie: curie of the node
        endpoint: endpoint of the synonymizer
        att_to_get: attribute to get from API
    Returns:
        Corresponding curie
    """
    if not curie or pd.isna(curie):
        return None

    result = requests.get(f"{endpoint}/normalize", json={"names": [curie]})

    element = result.json().get(curie)
    if element:
        return element.get("id", {}).get(att_to_get)

    return None


def coalesce(s: pd.Series, *series: List[pd.Series]):
    """Coalesce the column information like a SQL coalesce."""
    for other in series:
        s = s.mask(pd.isnull, other)
    return s


def enrich_df(
    df: pd.DataFrame, endpoint: str, func: Callable, input_cols: str, target_col: str
) -> pd.DataFrame:
    """Function to resolve nodes of the nodes input dataset.

    Args:
        df: nodes dataframe
        endpoint: endpoint of the synonymizer
        func: func to call
        input_cols: input cols, cols are coalesced to obtain single column
        target_col: target col
    Returns:
        dataframe enriched with Curie column
    """
    # Coalesce input cols
    col = coalesce(*[df[col] for col in input_cols])

    # Apply enrich function and replace nans by empty space
    df[target_col] = col.apply(partial(func, endpoint=endpoint))

    return df


@has_schema(
    schema={
        "ID": "numeric",
        "name": "object",
        "curie": "object",
        "normalized_curie": "object",
    },
    allow_subset=True,
)
@primary_key(primary_key=["ID"])
def create_int_nodes(nodes: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Function to create a intermediate nodes dataset by filtering and renaming columns."""
    # Enrich curie with node synonymizer
    resolved = enrich_df(
        nodes, endpoint, resolve, input_cols=["name"], target_col="curie"
    )

    # Normalize curie, by taking corrected currie or curie
    normalized = enrich_df(
        resolved,
        endpoint,
        normalize,
        input_cols=["corrected_curie", "curie"],
        target_col="normalized_curie",
    )

    # If new id is specified, we use the new id as a new KG identifier should be introduced
    normalized["normalized_curie"] = coalesce(
        normalized["new_id"], normalized["normalized_curie"]
    )

    return normalized


@has_schema(
    schema={
        "SourceId": "object",
        "TargetId": "object",
    },
    allow_subset=True,
)
def create_int_edges(int_nodes: pd.DataFrame, int_edges: pd.DataFrame) -> pd.DataFrame:
    """Function to create int edges dataset.

    Function ensures edges dataset link curies in the KG.
    """
    # Remove all nodes that could not be resolved, as we wont include
    # any edges between those.
    index = int_nodes[int_nodes["normalized_curie"].notna()]

    res = (
        int_edges.merge(
            index.rename(columns={"normalized_curie": "SourceId"}),
            left_on="Source",
            right_on="ID",
            how="left",
        )
        .drop(columns="ID")
        .merge(
            index.rename(columns={"normalized_curie": "TargetId"}),
            left_on="Target",
            right_on="ID",
            how="left",
        )
        .drop(columns="ID")
    )

    res["Included"] = res.apply(
        lambda row: not (pd.isna(row["SourceId"]) or pd.isna(row["TargetId"])), axis=1
    )

    return res


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
@primary_key(primary_key=["subject", "predicate", "object"])
def create_prm_edges(int_edges: pd.DataFrame) -> pd.DataFrame:
    """Function to create a primary edges dataset by filtering and renaming columns."""
    # Replace empty strings with nan
    res = int_edges.rename(
        columns={"SourceId": "subject", "TargetId": "object", "Label": "predicate"}
    ).dropna(subset=["subject", "object"])

    res["predicate"] = "biolink:" + res["predicate"]
    res["knowledge_source"] = "ec:medical"

    return res


@has_schema(
    schema={
        "clinical_trial_id": "object",
        "reason_for_rejection": "object",
        "drug_name": "object",
        "disease_name": "object",
        "significantly_better": "numeric",
        "non_significantly_better": "numeric",
        "non_significantly_worse": "numeric",
        "significantly_worse": "numeric",
    },
    allow_subset=True,
    df="df",
)
def map_name_to_curie(
    df: pd.DataFrame, endpoint: str, drug_types: List[str], disease_types: List[str]
) -> pd.DataFrame:
    """Map drug name to curie.

    Function to map drug name or disease name in raw clinical trail dataset to curie using the synonymizer.
    And check after mapping, if the mapped curies are the same in different rows, we check whether their the
    clinical performance is the same. If not, we label them as "True" in the "conflict" column, otherwise "False".

    Args:
        df: raw clinical trial dataset from medical team
        endpoint: endpoint of the synonymizer
        drug_types: list of drug types
        disease_types: list of disease types
    Returns:
        dataframe with two additional columns: "Mapped Drug Curie" and "Mapped Drug Disease"
    """
    # Map the drug name to the corresponding curie ids
    df["drug_kg_curie"] = df["drug_name"].apply(
        lambda x: normalize(x, endpoint=endpoint)
    )
    df["drug_kg_label"] = df["drug_name"].apply(
        lambda x: normalize(x, endpoint=endpoint, att_to_get="category")
    )

    # Map the disease name to the corresponding curie ids
    df["disease_kg_curie"] = df["disease_name"].apply(
        lambda x: normalize(x, endpoint=endpoint)
    )
    df["disease_kg_label"] = df["disease_name"].apply(
        lambda x: normalize(x, endpoint=endpoint, att_to_get="category")
    )

    # Validate correct labels
    df["label_included"] = (df["drug_kg_label"].isin(drug_types)) & (
        df["disease_kg_label"].isin(disease_types)
    )

    # check conflict
    df["conflict"] = (
        df.groupby(["drug_kg_curie", "disease_kg_curie"])[
            [
                "significantly_better",
                "non_significantly_better",
                "non_significantly_worse",
                "significantly_worse",
            ]
        ]
        .transform(lambda x: x.nunique() > 1)
        .any(axis=1)
    )

    return df


@has_schema(
    schema={
        "clinical_trial_id": "object",
        "reason_for_rejection": "object",
        "drug_name": "object",
        "disease_name": "object",
        "drug_kg_curie": "object",
        "disease_kg_curie": "object",
        "conflict": "object",
        "significantly_better": "numeric",
        "non_significantly_better": "numeric",
        "non_significantly_worse": "numeric",
        "significantly_worse": "numeric",
    },
    allow_subset=True,
    df="df",
)
@primary_key(
    primary_key=[
        "clinical_trial_id",
        "drug_kg_curie",
        "disease_kg_curie",
    ]
)
def clean_clinical_trial_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean clinical trails data.

    Function to clean the mapped clinical trial dataset for use in time-split evaluation metrics.

    Args:
        df: raw clinical trial dataset added with mapped drug and disease curies
    Returns:
        Cleaned clinical trial data.
    """
    # Remove rows with conflicts
    df = df[df["conflict"].eq("FALSE")].reset_index(drop=True)

    # remove rows with reason for rejection
    df = df[df["reason_for_rejection"].isna()].reset_index(drop=True)

    # Define columns to check
    columns_to_check = [
        "drug_kg_curie",
        "disease_kg_curie",
        "significantly_better",
        "non_significantly_better",
        "non_significantly_worse",
        "significantly_worse",
    ]

    # Remove rows with missing values in cols
    df = df.dropna(subset=columns_to_check).reset_index(drop=True)

    # drop columns
    df = df.drop(columns=["reason_for_rejection", "conflict"]).reset_index(drop=True)

    return df


@has_schema(
    schema={"single_ID": "object", "curie": "object", "name": "object"},
    allow_subset=True,
)
# @primary_key(primary_key=["single_ID"]) #TODO: re-introduce once the drug list is ready
def clean_drug_list(drug_df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Synonymize the drug list and filter out NaNs.

    Args:
        drug_df: disease list in a dataframe format.
        endpoint: endpoint of the synonymizer.

    Returns:
        dataframe with synonymized drug IDs in normalized_curie column.
    """
    res = enrich_df(
        drug_df,
        func=normalize,
        input_cols=["single_ID"],
        target_col="curie",
        endpoint=endpoint,
    )

    res = enrich_df(
        res,
        func=partial(normalize, att_to_get="name"),
        input_cols=["single_ID"],
        target_col="name",
        endpoint=endpoint,
    )
    return res.loc[~res["curie"].isna()]


@has_schema(
    schema={
        "category_class": "object",
        "label": "object",
        "definition": "object",
        "synonyms": "object",
        "subsets": "object",
        "crossreferences": "object",
        "curie": "object",
        "name": "object",
    },
    allow_subset=True,
)
@primary_key(primary_key=["category_class", "curie"])
def clean_disease_list(disease_df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Synonymize the disease list and filter out NaNs.

    Args:
        disease_df: disease list in a dataframe format.
        endpoint: endpoint of the synonymizer.

    Returns:
        dataframe with synonymized disease IDs in normalized_curie column.
    """
    res = enrich_df(
        disease_df,
        func=normalize,
        input_cols=["category_class"],
        target_col="curie",
        endpoint=endpoint,
    )
    res = enrich_df(
        res,
        func=partial(normalize, att_to_get="name"),
        input_cols=["category_class"],
        target_col="name",
        endpoint=endpoint,
    )
    return res.loc[~res["curie"].isna()]
