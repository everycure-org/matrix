import networkx as nx
import numpy as np
import pandas as pd
from data_fabricator.v0.nodes.fabrication import fabricate_datasets
from kedro.pipeline import Pipeline, node, pipeline

from matrix.kedro4argo_node import ArgoNode


def _create_random_pairs(
    drug_list: pd.DataFrame,
    disease_list: pd.DataFrame,
    num: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Create random drug-disease pairs. Ensures no duplicate pairs.

    Args:
        drug_list: Dataframe containing the list of drugs.
        disease_list: Dataframe containing the list of diseases.
        num: Size of each set of random pairs. Defaults to 100.
        seed: Random seed. Defaults to 42.

    Returns:
        Dataframe containing unique drug-disease pairs.
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
            columns=["source", "target", "drug|disease"],
        )

        # Remove duplicate pairs
        df = df.drop_duplicates()

        # Check that we still have enough fabricated pairs
        is_enough_generated = len(df) >= 2 * num or attempt > 100
        attempt += 1

    return df[: 2 * num]


def _create_kgml_xdtd_gtpairs(
    drug_list: pd.DataFrame,
    disease_list: pd.DataFrame,
    num: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 2 sets of random drug-disease pairs. Wrapper for _create_random_pairs.

    Returns:
        Two dataframes, each containing 'num' unique drug-disease pairs.
    """
    df = _create_random_pairs(drug_list, disease_list, num, seed)
    return df[:num], df[num : 2 * num]


def _create_ec_gt_pairs(
    drug_list: pd.DataFrame,
    disease_list: pd.DataFrame,
    num: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 2 sets of random drug-disease pairs with additional metadata. Wrapper for _create_random_pairs.

    Returns:
        Two dataframes containing unique drug-disease pairs with metadata.
    """
    df = _create_random_pairs(drug_list, disease_list, num, seed)
    positives = df[:num]
    negatives = df[num : 2 * num]

    BOOL_COLS = ["is_diagnostic_agent", "is_allergen", "llm_nameres_correct", "llm_nameres_correct_drug"]
    STRING_COLS = [
        "final normalized drug label",
        "final normalized disease label",
        "active ingredient",
        "contraindications",
        "disease contraindicated",
        "disease id nameres",
        "disease label nameres",
        "llm_nameres_correct",
        "llm disease id",
        "drug id nameres",
        "drug label nameres",
        "final normalized disease id",
        "final normalized disease label",
        "final normalized drug label",
        "llm drug id",
    ]
    # Rename columns
    positives = positives.rename(
        columns={"source": "final normalized drug id", "target": "final normalized disease id"}
    )
    negatives = negatives.rename(
        columns={"source": "final normalized drug id", "target": "final normalized disease id"}
    )

    for col in STRING_COLS:
        negatives[col] = np.random.choice(["string", "dummy"], len(negatives), p=[0.3, 0.7])

    for col in BOOL_COLS:
        negatives[col] = np.random.choice([True, False], len(negatives), p=[0.3, 0.7])

    return positives, negatives


def generate_paths(edges: pd.DataFrame, positives: pd.DataFrame, negatives: pd.DataFrame):
    def find_path(graph, start, end):
        try:
            # Find the shortest path between start and end
            path = nx.shortest_path(graph, source=start, target=end)
            return [
                {
                    "source": path[i],
                    "target": path[i + 1],
                    "key": graph.get_edge_data(path[i], path[i + 1])["predicate"],
                }
                for i in range(len(path) - 1)
            ]
        except Exception:
            return None

    graph = nx.DiGraph()

    # Fill graph
    for _, row in edges.iterrows():
        graph.add_edge(row["subject"], row["object"], predicate=row["predicate"])

    # Generate paths for GT
    rows = []
    ground_truth = pd.concat([positives, negatives])
    for idx, row in ground_truth.iterrows():
        if path := find_path(graph, row["source"], row["target"]):
            rows.append({"graph": {"_id": str(idx)}, "links": path})

    return rows


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
                    "disease_list": "ingestion.raw.disease_list",
                    "drug_list": "ingestion.raw.drug_list",
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
            node(
                func=_create_kgml_xdtd_gtpairs,
                inputs=[
                    "ingestion.raw.drug_list",
                    "ingestion.raw.disease_list",
                ],
                outputs=[
                    "ingestion.raw.ground_truth.positives",
                    "ingestion.raw.ground_truth.negatives",
                ],
                name="create_gt_pairs",
            ),
            node(
                func=_create_ec_gt_pairs,
                inputs=[
                    "ingestion.raw.drug_list",
                    "ingestion.raw.disease_list",
                ],
                outputs=[
                    "ingestion.raw.ec_ground_truth.positives",
                    "ingestion.raw.ec_ground_truth.negatives",
                ],
                name="create_ec_gt_pairs",
            ),
            node(
                func=_create_ec_gt_pairs,
                inputs=[
                    "ingestion.raw.drug_list",
                    "ingestion.raw.disease_list",
                ],
                outputs=[
                    "ingestion.raw.ec_ground_truth_downfilled.positives",
                    "ingestion.raw.ec_ground_truth_downfilled.negatives",
                ],
                name="create_ec_gt_pairs_downfilled",
            ),
            node(
                func=generate_paths,
                inputs=[
                    "ingestion.raw.rtx_kg2.edges@pandas",
                    "ingestion.raw.ground_truth.positives",
                    "ingestion.raw.ground_truth.negatives",
                ],
                outputs="ingestion.raw.drugmech.edges@pandas",
                name="create_drugmech_pairs",
            ),
        ]
    )
