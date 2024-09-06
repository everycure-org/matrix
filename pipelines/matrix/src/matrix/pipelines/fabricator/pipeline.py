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
                    "clinical_trials": "ingestion.raw.clinical_trials_data",
                },
                name="fabricate_kg2_datasets",
            ),
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.ec_medical_kg"},
                outputs={
                    "nodes": "ingestion.raw.ec_medical_team.nodes@pandas",
                    "edges": "ingestion.raw.ec_medical_team.edges@pandas",
                },
                name="fabricate_ec_medical_datasets",
            ),
            node(
                func=_create_pairs,
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="modelling.raw.ground_truth.positives",
                name="create_tp_pairs",
            ),
            node(
                func=_create_pairs,
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="modelling.raw.ground_truth.negatives",
                name="create_tn_pairs",
            ),
        ]
    )
