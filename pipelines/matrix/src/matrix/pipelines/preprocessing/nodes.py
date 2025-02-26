import json
import logging
import random
import time
from typing import Iterable, List, Tuple

import pandas as pd
import requests
from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output
from tenacity import Retrying, retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def coalesce(s: pd.Series, *series: List[pd.Series]):
    """Coalesce the column information like a SQL coalesce."""
    for other in series:
        s = s.mask(pd.isnull, other)
    return s


def resolve_one_name_batch(names: List[str], url: str) -> dict:
    """Batch resolve a list of names to their corresponding CURIEs."""
    payload = {
        "strings": names,
        "autocomplete": True,
        "highlighting": False,
        "offset": 0,
        "limit": 1,
    }
    # Waiting between requests drastically improves the API performance, as opposed to hitting a 5xx code
    # and using retrying with backoff, which can render the API unresponsive for a long time (> 10 min).
    time.sleep(random.randint(5, 10))
    for attempt in Retrying(wait=wait_exponential(multiplier=2, min=2, max=120), stop=stop_after_attempt(5)):
        with attempt:
            response = requests.post(url, json=payload)
            logger.debug(f"Request time: {response.elapsed.total_seconds():.2f} seconds")
            response.raise_for_status()
    return response.json(), response.elapsed.total_seconds()


def parse_one_name_batch(
    result: dict[str, list[dict[str, str]]], cols_to_get: Iterable[str]
) -> dict[str, dict[str, str | None]]:
    """Parse API response to extract resolved names and corresponding attributes."""
    resolved_data = {}

    for name, attributes in result.items():
        if attributes:
            resolved_data[name] = {col: attributes[0].get(col) for col in cols_to_get}
        else:
            resolved_data[name] = dict.fromkeys(cols_to_get)

    return resolved_data


def resolve_names(names: str, cols_to_get: Iterable[str], url: str, batch_size: int) -> dict:
    """Function to retrieve the normalized identifier through the normalizer.

    Args:
        name: name of the node to be resolved
        cols_to_get: attribute to get from API
    Returns:
        Name and corresponding curie
    """

    resolved_data = {}
    tot_elapsed = 0
    iterations = 0
    for i in range(0, len(names), batch_size):
        logger.debug(f"Running batch {iterations} of size {batch_size}")
        batch = names[i : i + batch_size]
        batch_response, elapsed = resolve_one_name_batch(batch, url)
        tot_elapsed += elapsed
        logger.debug(f"Running total elapsed: {tot_elapsed}")
        batch_parsed = parse_one_name_batch(batch_response, cols_to_get)
        resolved_data.update(batch_parsed)
        iterations += 1
    logger.debug(f"Total elapsed was: {tot_elapsed}")
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
    # return pd.read_pickle("process_medical_nodes_df_batch.pkl")

    start = time.perf_counter()
    logger.debug("Running process_medical_nodes")

    names = df["name"].dropna().unique().tolist()
    resolved_names = resolve_names(
        names, cols_to_get=["curie", "label", "types"], url=resolver_url, batch_size=batch_size
    )
    extra_cols = df.name.map(resolved_names)
    df = df.join(pd.json_normalize(extra_cols))

    # Coalesce id and new id to allow adding "new" nodes
    df["normalized_curie"] = coalesce(df["new_id"], df["curie"])

    # Filter out nodes that are not resolved
    is_resolved = df["normalized_curie"].notna()
    df = df[is_resolved]
    if not is_resolved.all():
        logger.warning(f"{(~is_resolved).sum()} EC medical nodes have not been resolved.")

    # Flag the number of duplicate IDs
    is_unique = df["normalized_curie"].groupby(df["normalized_curie"]).transform("count") == 1
    if not is_unique.all():
        logger.warning(f"{(~is_unique).sum()} EC medical nodes are duplicated.")

    # df.to_pickle("process_medical_nodes_df_batch.pkl")

    end = time.perf_counter()
    print(f"Execution time process_medical_nodes: {end - start:.6f} seconds")

    return df


@check_output(
    schema=DataFrameSchema(
        columns={
            "SourceId": Column(str, nullable=False),
            "TargetId": Column(str, nullable=False),
            "Label": Column(str, nullable=False),
        },
        unique=["SourceId", "TargetId", "Label"],
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
        unique=["drug_curie", "disease_curie"],
    )
)
def add_source_and_target_to_clinical_trails(df: pd.DataFrame, resolver_url: str, batch_size: int) -> pd.DataFrame:
    """Resolve names to curies for source and target columns in clinical trials data.

    Args:
        df: Clinical trial dataset
    """
    # return pd.read_pickle("add_source_and_target_df_batch.pkl")
    # dups = df[df.duplicated(subset=["drug_curie", "disease_curie"], keep=False)].sort_values('drug_curie')

    start = time.perf_counter()

    drug_names = df["drug_name"].dropna().unique().tolist()
    disease_names = df["disease_name"].dropna().unique().tolist()

    drug_mapping = resolve_names(drug_names, cols_to_get=["curie"], url=resolver_url, batch_size=batch_size)
    disease_mapping = resolve_names(disease_names, cols_to_get=["curie"], url=resolver_url, batch_size=batch_size)

    drug_mapping_df = pd.DataFrame(drug_mapping).T["curie"].rename("drug_curie")
    disease_mapping_df = pd.DataFrame(disease_mapping).T["curie"].rename("disease_curie")

    df = pd.merge(df, drug_mapping_df, how="left", left_on="drug_name", right_index=True)
    df = pd.merge(df, disease_mapping_df, how="left", left_on="disease_name", right_index=True)

    # df["drug_curie"] = df["drug_name"].map(lambda x: drug_mapping.get(x, {}).get("curie", None))
    # df["disease_curie"] = df["disease_name"].map(lambda x: disease_mapping.get(x, {}).get("curie", None))
    # df.to_pickle("add_source_and_target_df_batch.pkl")
    end = time.perf_counter()
    print(f"Execution time add_source_and_target_to_clinical_trails: {end - start:.6f} seconds")
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


# -------------------------------------------------------------------------
# Ground Truth Concatenation
# -------------------------------------------------------------------------


@check_output(
    schema=DataFrameSchema(
        columns={
            "drug|disease": Column(str, nullable=False),
            "y": Column(int, nullable=False),
        },
        unique=["source", "target", "drug|disease"],
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
