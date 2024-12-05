from typing import List

import pandas as pd
from data_fabricator.v0.nodes.fabrication import fabricate_datasets
from kedro.pipeline import Pipeline, node, pipeline

import networkx as nx


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
        random_diseases = disease_list["curie"].sample(num * 4, replace=True, ignore_index=True, random_state=2 * seed)

        df = pd.DataFrame(
            data=[[drug, disease] for drug, disease in zip(random_drugs, random_diseases)],
            columns=["source", "target"],
        )

        # Remove duplicate pairs
        df = df.drop_duplicates()

        # Check that we still have enough fabricated pairs
        is_enough_generated = len(df) >= num or attempt > 100
        attempt += 1

    return df[:num], df[num : 2 * num]


def expand_paths(node: str, graph: nx.DiGraph, length: int, path: List[str]):
    """Function to expand paths from given node."""
    if length == 0:
        return [*path, node]

    # Expand children
    return [
        expand_paths(successor, graph, length - 1, [*path, node]) for successor in frozenset(graph.successors(node))
    ]


def generate_random_paths(edges: pd.DataFrame, depth: int = 2, seed: int = 42):
    """Function to generate dataframe with random paths."""
    # Initialize a GraphX instance
    graph = nx.DiGraph()
    for _, row in edges.iterrows():
        graph.add_edge(row["subject"], row["object"])

    # Generate paths
    nodes = set(edges["subject"].sample(1, replace=True, ignore_index=True, random_state=seed).tolist())
    paths = []

    # Expand path
    for el in nodes:
        for expand_path in expand_paths(el, graph, depth, []):
            paths.extend(expand_path)

    df = pd.DataFrame({"path": paths})
    df["index"] = df.index
    return df


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
                    "pubmed_ids_mapping": "ingestion.raw.rtx_kg2.curie_to_pmids@pandas",
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
                    "modelling.raw.ground_truth.positives@pandas",
                    "modelling.raw.ground_truth.negatives@pandas",
                ],
                name="create_gn_pairs",
            ),
            node(
                func=generate_random_paths,
                inputs=[
                    "ingestion.raw.rtx_kg2.edges@pandas",
                ],
                outputs="ingestion.pre.drugmech.paths@pandas",
                name="generate_drugmech_paths",
            ),
        ]
    )
