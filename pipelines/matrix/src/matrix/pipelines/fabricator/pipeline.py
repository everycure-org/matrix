"""Fabricator pipeline."""

import pandas as pd
import pyspark.sql as sql

from kedro.pipeline import Pipeline, node, pipeline

from data_fabricator.v0.nodes.fabrication import fabricate_datasets

from pyspark.sql import DataFrame  # TODO: REMOVE?

import pyspark.sql.functions as F


def _create_pairs(
    drug_list: pd.DataFrame, disease_list: pd.DataFrame, num: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Creating 2 sets of random pairs from the nodes. Ensures no duplicate pairs."""
    # Sample random pairs
    random_drugs = drug_list["curie"].sample(
        num * 2, replace=True, ignore_index=True, random_state=seed
    )
    random_diseases = disease_list["curie"].sample(
        num * 2, replace=True, ignore_index=True, random_state=seed
    )

    df = pd.DataFrame(
        data=[[drug, disease] for drug, disease in zip(random_drugs, random_diseases)],
        columns=["source", "target"],
    )

    # Remove duplicate pairs
    df = df.drop_duplicates()

    # split df in half and return two df
    return df[:num], df[num:]


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
                inputs=[
                    "ingestion.raw.drug_list@pandas",
                    "ingestion.raw.disease_list@pandas",
                ],
                outputs=[
                    "modelling.raw.ground_truth.positives",
                    "modelling.raw.ground_truth.negatives",
                ],
                name="create_gn_pairs",
            ),
        ]
    )
