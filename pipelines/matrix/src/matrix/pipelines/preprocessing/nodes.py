import logging
import random
import time
from typing import Collection, Dict, List, Sequence

import pandas as pd
import pyspark.sql.functions as f
import requests
from matrix.pipelines.preprocessing.normalization import resolve_ids_batch_async
from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType, StringType
from tenacity import Retrying, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

import asyncio
from typing import Dict, List

# -------------------------------------------------------------------------
# Embiology Dataset
# -------------------------------------------------------------------------


def normalize_identifiers(attr: pd.DataFrame, norm_params: Dict[str, str]) -> pd.DataFrame:
    """Normalize identifiers in embiology attributes using NCATS normalizer.

    Args:
        attr: embiology attributes
        identifiers_mapping: mapping of identifiers

    Returns:
        pd.DataFrame: DataFrame with normalized identifiers
    """
    # Get values list from attribute DataFrame
    ids_to_resolve = attr.select("value", "id").distinct().rdd.map(lambda x: (x.value, x.id)).collectAsMap()
    # Run async resolution
    results = asyncio.run(
        resolve_ids_batch_async(
            curies=list(ids_to_resolve.keys()),
            batch_size=norm_params["batch_size"],
            max_concurrent=norm_params["max_concurrent"],
            url=norm_params["url"],
        )
    )
    # Convert results to pandas DataFrame
    results_df = pd.DataFrame(
        [
            {
                "curie": k,
                "id": v["id"],
                "label": v["label"],
                "all_categories": v["all_categories"],
                "equivalent_identifiers": v["equivalent_identifiers"],
                "original_id": v["original_id"],
            }
            for k, v in results.items()
        ]
    )
    results_df["attribute_id"] = results_df["original_id"].map(ids_to_resolve)
    return results_df


def prepare_normalized_identifiers(
    attr: pd.DataFrame,
    nodes: pd.DataFrame,
    manual_id_mapping: pd.DataFrame,
    manual_name_mapping: pd.DataFrame,
    identifiers_mapping: Dict[str, str],
    norm_params: Dict[str, str],
) -> pd.DataFrame:
    """Prepare identifiers for embiology nodes & normalize using NCATS normalizer.

    Args:
        attr: embiology attributes
        identifiers_mapping: mapping of identifiers
    """

    # Create a UDF for the mapping lookup
    def lookup_mapping(name):
        return identifiers_mapping.get(name, "")

    lookup_udf = udf(lookup_mapping, StringType())

    df = attr.withColumn(
        "value",
        f.when(
            f.col("name").isin(list(identifiers_mapping.keys())),
            f.concat(
                lookup_udf(f.col("name")),
                f.split(f.col("value"), "\.").getItem(0),
            ),
        ).otherwise(f.col("value")),
    )
    df = df.withColumn(
        "normalized_id",
        f.when(f.col("name").isin(list(identifiers_mapping.keys())), f.lit(None)).otherwise(f.col("value")),
    )
    # Add manual ID mapping
    df_v0 = (
        attr.filter(f.col("name") == "MedScan ID")
        .join(
            manual_id_mapping.select("id", "identifier")
            .withColumnRenamed("identifier", "value")
            .withColumnRenamed("id", "curie"),
            on="value",
            how="inner",
        )
        .dropDuplicates()
        .withColumn("normalized_id", f.coalesce(f.col("curie"), f.col("value")))
        .select("id", "name", "value", "normalized_id")
    )
    df_v1 = (
        manual_name_mapping.join(nodes.select("name", "attributes"), on="name", how="inner")
        .withColumn("attributes", f.split(f.regexp_replace("attributes", "[{}]", ""), ","))
        .withColumn("attributes", f.explode(f.col("attributes")))
        .select("attributes", "id")
        .withColumn("name", f.lit("Manual ID"))
        .withColumn("value", f.col("id"))
        .select("attributes", "name", "value", "id")
    )
    df_concat = (
        df.filter(~f.col("id").isin(df_v0.select("id").rdd.map(lambda x: x.id).collect())).union(df_v0).union(df_v1)
    )
    output_df = normalize_identifiers(df_concat, norm_params)

    return output_df


def prepare_nodes(nodes, id_mapping, biolink_mapping):
    """Prepare nodes for embiology nodes.

    Args:
        nodes: embiology nodes
        identifiers_mapping: mapping of identifiers
    """
    # Normalize identifier
    norm_nodes = (
        nodes.withColumn("attributes", f.split(f.regexp_replace("attributes", "[{}]", ""), ","))
        .withColumn("attributes", f.explode(f.col("attributes")))
        .withColumnRenamed("attributes", "attribute_id")
        .join(
            id_mapping.withColumnRenamed("id", "normalized_id")
            .withColumn("normalization_success", f.col("normalized_id").isNotNull())
            .filter(f.col("normalization_success") == True),
            on="attribute_id",
            how="left",
        )
        .orderBy("normalization_success", ascending=False)
        .groupBy(["id", "urn", "name", "nodetype"])
        .agg(
            f.first(f.col("normalized_id"), ignorenulls=True).alias("normalized_id"),
            f.first(f.col("equivalent_identifiers"), ignorenulls=True).alias("equivalent_identifiers"),
            f.first(f.col("all_categories"), ignorenulls=True).alias("all_categories"),
            f.first(f.col("normalization_success"), ignorenulls=True).alias("normalization_success"),
        )
        .withColumn("final_id", f.coalesce(f.col("normalized_id"), f.col("urn")))
    )

    # Biolink fix
    def lookup_mapping(name):
        return biolink_mapping.get(name, "")

    lookup_udf = udf(lookup_mapping, StringType())
    norm_nodes = norm_nodes.withColumn("category", lookup_udf(f.col("nodetype")))
    return norm_nodes


def prepare_edges(control, attributes, biolink_mapping):
    """Prepare edges for embiology nodes.

    Args:
        nodes: embiology nodes
        identifiers_mapping: mapping of identifiers
    """
    # Fix directed and undirected edges
    control_directed = control.filter((f.col("inkey") != "{}") & (f.col("inoutkey") == "{}"))
    control_undirected = control.filter((f.col("inkey") == "{}") & (f.col("inoutkey") != "{}"))
    control_undirected = (
        control_undirected.withColumn("inoutkey", f.split(f.regexp_replace("inoutkey", "[{}]", ""), ","))
        .withColumn("inkey", f.col("inoutkey").getItem(0))
        .withColumn("outkey", f.col("inoutkey").getItem(1))
    )
    # Divide each undirected edge into two directed edges
    control_undirected = control_undirected.union(
        control_undirected.withColumnRenamed("inkey", "outkey_new")
        .withColumnRenamed("outkey", "inkey_new")
        .withColumnRenamed("outkey_new", "outkey")
        .withColumnRenamed("inkey_new", "inkey")
        .drop("outkey_new", "inkey_new")
        .select(control_undirected.columns)
    )
    # Union directed and undirected edges
    edges = (
        control_directed.withColumn("inoutkey", f.split(f.regexp_replace("inoutkey", "[{}]", ""), ","))
        .withColumn("inkey", f.regexp_replace("inkey", "[{}]", ""))
        .withColumn("outkey", f.regexp_replace("outkey", "[{}]", ""))
        .union(control_undirected)
    )

    # Biolink fix
    def lookup_mapping(name):
        return biolink_mapping.get(name, "")

    lookup_udf = udf(lookup_mapping, StringType())
    edges = edges.withColumn("predicate", lookup_udf(f.col("controltype")))

    # Add attributes
    edges = edges.join(attributes.withColumnRenamed("id", "attributes"), on="attributes", how="left")
    return edges


def add_edge_attributes(references):
    """Prepare edges for embiology nodes.

    Args:
        nodes: embiology nodes
        identifiers_mapping: mapping of identifiers
    """
    # Add Num_sentences & Num_references
    return (
        references.withColumn(
            "pmid",
            f.udf(lambda x: [f"PMID:{str(x)}"] if x is not None else None, ArrayType(StringType()))(f.col("pmid")),
        )
        .withColumn(
            "doi", f.udf(lambda x: [f"DOI:{str(x)}"] if x is not None else None, ArrayType(StringType()))(f.col("doi"))
        )
        .groupby("id")
        .agg(
            f.count("id").alias("num_sentences"),
            f.count_distinct("unique_ref").alias("num_references"),
            f.flatten(f.collect_set("pmid")).alias("pmid"),
            f.flatten(f.collect_set("doi")).alias("doi"),
        )
        .withColumn("publications", f.flatten(f.array(f.col("pmid"), f.col("doi"))))
        .drop("pmid", "doi")
    )


def deduplicate_and_clean(nodes, edges):
    """Deduplicate edges.

    Args:
        edges: edges
    """
    nodes = (
        nodes.withColumn("original_identifier", f.col("id").cast(StringType()))
        .drop("id")
        .withColumnRenamed("urn", "original_urn")
        .withColumnRenamed("nodetype", "original_nodetype")
        .withColumnRenamed("final_id", "id")
        .withColumn("equivalent_identifiers", f.array_join(f.col("equivalent_identifiers"), "|"))
        .withColumn("all_categories", f.array_join(f.col("all_categories"), "|"))
        .select(
            "id",
            "name",
            "category",
            "equivalent_identifiers",
            "all_categories",
            "original_identifier",
            "original_urn",
            "original_nodetype",
        )
    )
    # Get unique keys from edges using a union of inkey and outkey to ensure there are no duplicates
    edge_keys = (
        edges.select("inkey").union(edges.select("outkey")).distinct().withColumnRenamed("inkey", "original_identifier")
    )
    # Filter nodes to only keep those that appear in edges
    f_nodes = nodes.join(edge_keys, "original_identifier", "inner")
    f_nodes = f_nodes.dropDuplicates(["id"])
    f_nodes = nodes
    # Filter edges to only keep those where both subject and object nodes exist
    valid_identifiers = f_nodes.select("original_identifier").distinct()
    f_edges = edges.join(
        valid_identifiers.withColumnRenamed("original_identifier", "inkey"), on="inkey", how="inner"
    ).join(valid_identifiers.withColumnRenamed("original_identifier", "outkey"), on="outkey", how="inner")

    # Modify edge schema
    f_edges = (
        edges.withColumnRenamed("id", "original_id")
        .withColumn("original_subject", f.trim(f.col("inkey")))
        .withColumn("original_object", f.trim(f.col("outkey")))
        .withColumn("publications", f.array_join(f.col("publications"), "|"))
        .join(
            f_nodes.select("id", "original_identifier")
            .withColumnRenamed("original_identifier", "original_subject")
            .withColumnRenamed("id", "subject"),
            on="original_subject",
            how="left",
        )
        .join(
            f_nodes.select("id", "original_identifier")
            .withColumnRenamed("original_identifier", "original_object")
            .withColumnRenamed("id", "object"),
            on="original_object",
            how="left",
        )
    )
    return f_nodes.dropDuplicates(["id"]), f_edges.dropDuplicates(["subject", "object", "predicate"])


# -------------------------------------------------------------------------
# EC Clinical Data
# -------------------------------------------------------------------------


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


# -------------------------------------------------------------------------
# EC Medical Team Data
# -------------------------------------------------------------------------


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
