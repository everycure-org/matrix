"""Fabricator pipeline."""
import pandas as pd
import pyspark.sql as sql

from kedro.pipeline import Pipeline, node, pipeline

from data_fabricator.v0.nodes.fabrication import fabricate_datasets


def _create_pairs(nodes: sql.DataFrame, num: int = 50) -> pd.DataFrame:
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
                },
                name="fabricate_datasets",
            ),

            # borrowing inspiration from above, the `nodes` and `edges`
            # dict keys above correspond as defined in the fabricator parameters.yml,
            # while the values map the results to a catalog entry. We're generating
            # pandas dataframes, while the ingestion pipeline uses spark, therefore
            # we use the @pandas transcoding syntax.
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.robokop"},
                outputs={
                    "nodes": "ingestion.raw.robokop.nodes@pandas",
                    "edges": "ingestion.raw.robokop.edges@pandas",
                },
                name="fabricate_robokop",
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
        ]
    )
