"""Fabricator pipeline."""
import pandas as pd
import pyspark.sql as sql

from kedro.pipeline import Pipeline, node, pipeline

from data_fabricator.v0.nodes.fabrication import fabricate_datasets

from pyspark.sql import DataFrame

import pyspark.sql.functions as F


def _create_pairs(nodes: DataFrame, num: int = 50) -> pd.DataFrame:
    # NOTE: This is here because the dataset is generated without
    # header as per the KG2 schema. The spark version of the
    # dataset re-introduces the correct schema.
    nodes = nodes.toPandas()

    # Sample random pairs
    random_drugs = nodes["id"].sample(num, replace=True, ignore_index=True)
    random_diseases = nodes["id"].sample(num, replace=True, ignore_index=True)

    return pd.DataFrame(
        data=[[drug, disease] for drug, disease in zip(random_drugs, random_diseases)],
        columns=["source", "target"],
    )


def _edges_subset(edges: DataFrame, num: int = 10) -> pd.DataFrame:
    return edges.limit(num).withColumn("knowledge_source", F.lit("EveryCure"))


def create_pipeline(**kwargs) -> Pipeline:
    """Create fabricator pipeline."""
    return pipeline(
        [
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.rtx_kg2"},
                outputs={
                    "nodes": "ingestion.raw.rtx_kg2.nodes@pandas",
                    "edges": "ingestion.raw.rtx_kg2.edges@pandas",
                    "trails": "ingestion.raw.clinical_trial_data@pandas",
                },
                name="fabricate_datasets",
            ),
            node(
                func=_create_pairs,
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="integration.raw.ground_truth.positives",
                name="create_tp_pairs",
            ),
            node(
                func=_create_pairs,
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="integration.raw.ground_truth.negatives",
                name="create_tn_pairs",
            ),
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="ingestion.raw.ec_medical_team.nodes@spark",
                name="create_exp_nodes",
            ),
            # NOTE: Quickly taking a subset
            node(
                func=_edges_subset,
                inputs=["ingestion.raw.rtx_kg2.edges@spark"],
                outputs="ingestion.raw.ec_medical_team.edges@spark",
                name="create_exp_edges",
            ),
        ]
    )
