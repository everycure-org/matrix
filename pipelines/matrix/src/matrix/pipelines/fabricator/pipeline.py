"""Fabricator pipeline."""
import pandas as pd

from kedro.pipeline import Pipeline, node, pipeline

from data_fabricator.v0.nodes.fabrication import fabricate_datasets


def _create_pairs(nodes: pd.DataFrame, num: int = 50) -> pd.DataFrame:
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
                    "nodes": "modelling.raw.rtx_kg2.nodes",
                    "edges": "modelling.raw.rtx_kg2.edges",
                    "embeddings": "embeddings.int.graphsage",
                },
                name="fabricate_datasets",
            ),
            node(
                func=_create_pairs,
                inputs=["modelling.raw.rtx_kg2.nodes"],
                outputs="modelling.raw.ground_truth.tp",
                name="create_tp_pairs",
            ),
            node(
                func=_create_pairs,
                inputs=["modelling.raw.rtx_kg2.nodes"],
                outputs="modelling.raw.ground_truth.tn",
                name="create_tn_pairs",
            ),
        ]
    )
