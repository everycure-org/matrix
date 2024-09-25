"""Fabricator pipeline."""

import pandas as pd
import pyspark.sql as sql

from kedro.pipeline import Pipeline, node, pipeline

from data_fabricator.v0.nodes.fabrication import fabricate_datasets

from pyspark.sql import DataFrame

import pyspark.sql.functions as F


def _create_pairs(nodes: DataFrame, num: int = 50, seed: int = 42) -> pd.DataFrame:
    """Creating 2 sets of random pairs from the nodes.  Ensures no duplicate pairs.

    Args:
        nodes: Dataframe for fabricated nodes.
        num: Size of each set of random pairs. Defaults to 50.
        seed: Random seed. Defaults to 42.
    """
    # NOTE: This is here because the dataset is generated without
    # header as per the KG2 schema. The spark version of the
    # dataset re-introduces the correct schema.
    nodes = nodes.toPandas()

    is_enough_generated = False
    while not is_enough_generated:
        # Sample random pairs (we sample twice the required amount in case duplicates are removed)
        random_drugs = nodes["id"].sample(
            num * 4, replace=True, ignore_index=True, random_state=seed
        )
        random_diseases = nodes["id"].sample(
            num * 4, replace=True, ignore_index=True, random_state=2 * seed
        )

        df = pd.DataFrame(
            data=[
                [drug, disease] for drug, disease in zip(random_drugs, random_diseases)
            ],
            columns=["source", "target"],
        )

        # Remove duplicate pairs
        df = df.drop_duplicates()

        # Check that we still have enough fabricated pairs
        is_enough_generated = len(df) >= 2 * num

    # split df in half and return two df
    return df[:num], df[num : 2 * num]


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
                    "disease_list": "ingestion.raw.disease_list@pandas",
                    "drug_list": "ingestion.raw.drug_list@pandas",
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
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.robokop"},
                outputs={
                    "nodes": "ingestion.raw.robokop.nodes@pandas",
                    "edges": "ingestion.raw.robokop.edges@pandas",
                },
                name="fabricate_robokop_datasets",
            ),
            node(
                func=_create_pairs,
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs=[
                    "modelling.raw.ground_truth.positives",
                    "modelling.raw.ground_truth.negatives",
                ],
                name="create_gn_pairs",
            ),
        ]
    )
