import pandas as pd

from kedro.pipeline import Pipeline, node, pipeline

from data_fabricator.v0.nodes.fabrication import fabricate_datasets


def create_pairs(nodes: pd.DataFrame, num: int = 50) -> pd.DataFrame:
    # Sample random pairs
    random_drugs = nodes["id"].sample(num, replace=True, ignore_index=True)
    random_diseases = nodes["id"].sample(num, replace=True, ignore_index=True)

    return pd.DataFrame(
        data=[[drug, disease] for drug, disease in zip(random_drugs, random_diseases)],
        columns=["source", "target"],
    )


def create_fda_drugs(nodes: pd.DataFrame, num: int = 10) -> pd.DataFrame:
    return nodes["id"].sample(num, replace=True, ignore_index=True)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.rtx_kg2"},
                outputs={"nodes": "raw.rtx_kg2.nodes", "embeddings": "int.embeddings"},
                name="fabricate_datasets",
            ),
            node(
                func=create_pairs,
                inputs=["raw.rtx_kg2.nodes"],
                outputs="raw.ground_truth.tp",
                name="create_tp_pairs",
            ),
            node(
                func=create_pairs,
                inputs=["raw.rtx_kg2.nodes"],
                outputs="raw.ground_truth.tn",
                name="create_tn_pairs",
            ),
            # FUTURE: Move this transformation to pipeline
            node(
                func=create_fda_drugs,
                inputs=["raw.rtx_kg2.nodes"],
                outputs="raw.fda_drugs",
                name="create_fda_drugs",
            ),
        ]
    )
