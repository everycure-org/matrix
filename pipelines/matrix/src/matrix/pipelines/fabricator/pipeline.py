import pandas as pd
from data_fabricator.v0.nodes.fabrication import fabricate_datasets
from kedro.pipeline import Pipeline, node, pipeline

from matrix.kedro4argo_node import ArgoNode


def _create_pairs(
    drug_list: pd.DataFrame,
    disease_list: pd.DataFrame,
    num: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Create 2 sets of random drug-disease pairs. Ensures no duplicate pairs.

    Args:
        drug_list: Dataframe containing the list of drugs.
        disease_list: Dataframe containing the list of diseases.
        num: Size of each set of random pairs. Defaults to 100.
        seed: Random seed. Defaults to 42.

    Returns:
        Two dataframes, each containing 'num' unique drug-disease pairs.
    """
    is_enough_generated = False

    attempt = 0

    while not is_enough_generated:
        # Sample random pairs (we sample twice the required amount in case duplicates are removed)
        random_drugs = drug_list["curie"].sample(num * 4, replace=True, ignore_index=True, random_state=seed)
        random_diseases = disease_list["category_class"].sample(
            num * 4, replace=True, ignore_index=True, random_state=2 * seed
        )

        df = pd.DataFrame(
            data=[[drug, disease, f"{drug}|{disease}"] for drug, disease in zip(random_drugs, random_diseases)],
            columns=["subject", "object", "drug|disease"],
        )

        # Remove duplicate pairs
        df = df.drop_duplicates()

        # Check that we still have enough fabricated pairs
        is_enough_generated = len(df) >= num or attempt > 100
        attempt += 1
    tp_df = df[:num]
    tp_df["indication"] = True
    tp_df["contraindication"] = False
    tn_df = df[num : 2 * num]
    tn_df["indication"] = False
    tn_df["contraindication"] = True
    edges = pd.concat([tp_df, tn_df], ignore_index=True)
    id_list = set(edges.subject) | set(edges.object)
    nodes = pd.DataFrame(id_list, columns=["id"])
    return nodes, edges


def create_pipeline(**kwargs) -> Pipeline:
    """Create fabricator pipeline."""
    return pipeline(
        [
            ArgoNode(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.rtx_kg2"},
                outputs={
                    "nodes": "ingestion.raw.rtx_kg2.nodes@pandas",
                    "edges": "ingestion.raw.rtx_kg2.edges@pandas",
                    "disease_list": "ingestion.raw.disease_list.nodes@pandas",
                    "drug_list": "ingestion.raw.drug_list.nodes@pandas",
                    "pubmed_ids_mapping": "ingestion.raw.rtx_kg2.curie_to_pmids@pandas",
                },
                name="fabricate_kg2_datasets",
            ),
            ArgoNode(
                func=fabricate_datasets,
                inputs={
                    "fabrication_params": "params:fabricator.clinical_trials",
                    "rtx_nodes": "ingestion.raw.rtx_kg2.nodes@pandas",
                },
                outputs={
                    "nodes": "ingestion.raw.ec_clinical_trails.nodes@pandas",
                    "edges": "ingestion.raw.ec_clinical_trails.edges@pandas",
                },
                name="fabricate_clinical_trails_datasets",
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
            ArgoNode(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.robokop"},
                outputs={
                    "nodes": "ingestion.raw.robokop.nodes@pandas",
                    "edges": "ingestion.raw.robokop.edges@pandas",
                },
                name="fabricate_robokop_datasets",
            ),
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.spoke"},
                outputs={
                    "nodes": "ingestion.raw.spoke.nodes@pandas",
                    "edges": "ingestion.raw.spoke.edges@pandas",
                },
                name="fabricate_spoke_datasets",
            ),
            ArgoNode(
                func=_create_pairs,
                inputs=[
                    "ingestion.raw.drug_list.nodes@pandas",
                    "ingestion.raw.disease_list.nodes@pandas",
                ],
                outputs=[
                    "ingestion.raw.ground_truth.nodes@pandas",
                    "ingestion.raw.ground_truth.edges@pandas",
                ],
                name="create_gn_pairs",
            ),
        ]
    )
