from typing import List, Tuple

import pandas as pd
import requests
from matrix.utils.pa_utils import Column, DataFrameSchema, check_output
from tenacity import retry, stop_after_attempt, wait_exponential


def coalesce(s: pd.Series, *series: List[pd.Series]):
    """Coalesce the column information like a SQL coalesce."""
    for other in series:
        s = s.mask(pd.isnull, other)
    return s


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def resolve_name(name: str, cols_to_get: List[str]) -> dict:
    """Function to retrieve the normalized identifier through the normalizer.

    Args:
        name: name of the node to be resolved
        cols_to_get: attribute to get from API
    Returns:
        Name and corresponding curie
    """

    if not name or pd.isna(name):
        return {}

    result = requests.get(
        f"https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=True&highlighting=False&offset=0&limit=1"
    )
    if len(result.json()) != 0:
        element = result.json()[0]
        print({col: element.get(col) for col in cols_to_get})
        return {col: element.get(col) for col in cols_to_get}

    return {}


def process_medical_nodes(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize the name
    enriched_data = df["name"].apply(resolve_name, cols_to_get=["curie", "label", "types"])

    # Extract into df
    enriched_df = pd.DataFrame(enriched_data.tolist())
    df = pd.concat([df, enriched_df], axis=1)

    # Coalesce id and new id to allow adding "new" nodes
    df["normalized_curie"] = coalesce(df["new_id"], df["curie"])

    return df


@check_output(
    schema=DataFrameSchema(
        columns={
            "SourceId": Column(str, nullable=False),
            "TargetId": Column(str, nullable=False),
        }
    )
)
def process_medical_edges(int_nodes: pd.DataFrame, int_edges: pd.DataFrame) -> pd.DataFrame:
    """Function to create int edges dataset.

    Function ensures edges dataset link curies in the KG.
    """
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

    res["Included"] = res.apply(lambda row: not (pd.isna(row["SourceId"]) or pd.isna(row["TargetId"])), axis=1)

    return res


def add_source_and_target_to_clinical_trails(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve names to curies for source and target columns in clinical trials data.

    Args:
        df: Clinical trial dataset
    """
    # Normalize the name
    drug_data = df["drug_name"].apply(resolve_name, cols_to_get=["curie"])
    disease_data = df["disease_name"].apply(resolve_name, cols_to_get=["curie"])

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
            "significantly_better": Column(bool, nullable=False),
            "non_significantly_better": Column(bool, nullable=False),
            "non_significantly_worse": Column(bool, nullable=False),
            "significantly_worse": Column(bool, nullable=False),
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
    df = df[["drug_curie", "disease_curie", *name_columns, *outcome_columns]].astype(
        {col: bool for col in outcome_columns}
    )

    # Drop rows with missing values in cols
    df = df.dropna().reset_index(drop=True)

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
            if row[outcome_columns[n]] == True:
                row[outcome_columns[n + 1 :]] = False
        return row

    edges = edges.apply(ensure_one_true, axis=1)

    # Extract nodes
    drugs = df.rename(columns={"drug_curie": "curie", "drug_name": "name"})[["curie", "name"]]
    diseases = df.rename(columns={"disease_curie": "curie", "disease_name": "name"})[["curie", "name"]]
    nodes = pd.concat([drugs, diseases], ignore_index=True).drop_duplicates(subset="curie")
    return {"nodes": nodes, "edges": edges}


# -------------------------------------------------------------------------
# Ground Truth Concatenation
# -------------------------------------------------------------------------


@check_output(
    schema=DataFrameSchema(
        columns={
            "drug|disease": Column(str, nullable=False),
            "y": Column(int, nullable=False),
            # TODO: Piotr add
        },
        unique=["clinical_trial_id", "drug_kg_curie", "disease_kg_curie"],
    )
)
def create_gt(pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    """Converts the KGML-xDTD true positives and true negative dataframes into a singular dataframe compatible with EC format."""
    pos_df["indication"], pos_df["contraindication"] = True, False
    pos_df["y"] = 1
    neg_df["indication"], neg_df["contraindication"] = False, True
    neg_df["y"] = 0
    gt_df = pd.concat([pos_df, neg_df], axis=0)
    gt_df["drug|disease"] = gt_df["source"] + "|" + gt_df["target"]
    return gt_df


def create_gt_nodes_edges(edges: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    id_list = set(edges.source) | set(edges.target)
    nodes = pd.DataFrame(id_list, columns=["id"])
    edges.rename({"source": "subject", "target": "object"}, axis=1, inplace=True)
    return nodes, edges
