import itertools
import random

import networkx as nx
import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline
from matrix_fabricator.fabrication import fabricate_datasets

from . import nodes


def _create_pairs(
    drug_list: pd.DataFrame,
    disease_list: pd.DataFrame,
    num: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Creates 2 sets of random drug-disease pairs. Ensures no duplicate pairs.

    Args:
        drug_list: Dataframe containing the list of drugs.
        disease_list: Dataframe containing the list of diseases.
        num: Size of each set of random pairs.
        seed: Random seed.

    Returns:
        Two dataframes, each containing 'num' unique drug-disease pairs.
    """
    random.seed(seed)

    # Convert lists to sets of unique ids
    drug_ids = list(drug_list["id"].unique())
    disease_ids = list(disease_list["id"].unique())

    # Check that we have enough pairs
    if not len(drug_ids) * len(disease_ids) >= 2 * num:
        raise ValueError("Drug and disease lists are too small to generate the required number of pairs")

    # Subsample the lists to reduce memory usage
    for entity_list in [drug_ids, disease_ids]:
        if len(entity_list) > 2 * num:
            entity_list = random.sample(entity_list, 2 * num)

    # Create pairs and sample without replacement to ensure no duplicates
    pairs = list(itertools.product(drug_ids, disease_ids))
    df_sample = pd.DataFrame(random.sample(pairs, 2 * num), columns=["source", "target"])

    # Split into positives and negatives
    return df_sample[:num], df_sample[num:]


def remove_overlap(disease_list: pd.DataFrame, drug_list: pd.DataFrame):
    """Function to ensure no overlap between drug and disease lists.

    Due to our generator setup, it's possible our drug and disease sets
    are not disjoint.

    Args:
        drug_list: Dataframe containing the list of drugs.
        disease_list: Dataframe containing the list of diseases.

    Returns:
        Two dataframes, clean drug and disease lists respectively.
    """
    overlap = set(disease_list["id"]).intersection(set(drug_list["id"]))
    overlap_mask_drug = drug_list["id"].isin(overlap)
    overlap_mask_disease = disease_list["id"].isin(overlap)
    drug_list = drug_list[~overlap_mask_drug]
    disease_list = disease_list[~overlap_mask_disease]

    return {"disease_list": disease_list, "drug_list": drug_list}


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
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.rtx_kg2"},
                outputs={
                    "nodes": "ingestion.raw.rtx_kg2.nodes@pandas",
                    "edges": "ingestion.raw.rtx_kg2.edges@pandas",
                    "disease_list": "fabricator.int.disease_list",
                    "drug_list": "fabricator.int.drug_list",
                    "pubmed_ids_mapping": "ingestion.raw.rtx_kg2.curie_to_pmids@pandas",
                },
                name="fabricate_kg2_datasets",
            ),
            node(
                func=nodes.validate_datasets,
                inputs={"nodes": "ingestion.raw.rtx_kg2.nodes@polars", "edges": "ingestion.raw.rtx_kg2.edges@polars"},
                outputs="fabricator.int.rtx_kg2.violations",
                name="validate_fabricated_kg2_datasets",
            ),
            node(
                func=remove_overlap,
                inputs={
                    "disease_list": "fabricator.int.disease_list",
                    "drug_list": "fabricator.int.drug_list",
                },
                outputs={
                    "disease_list": "ingestion.raw.disease_list",
                    "drug_list": "ingestion.raw.drug_list",
                },
            ),
            node(
                func=fabricate_datasets,
                inputs={
                    "fabrication_params": "params:fabricator.clinical_trials.graph",
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
                inputs={
                    "fabrication_params": "params:fabricator.off_label.graph",
                    "rtx_nodes": "ingestion.raw.rtx_kg2.nodes@pandas",
                },
                outputs={
                    "nodes": "ingestion.raw.off_label.nodes@pandas",
                    "edges": "ingestion.raw.off_label.edges@pandas",
                },
                name="fabricate_off_label_datasets",
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
                func=nodes.validate_datasets,
                inputs={"nodes": "ingestion.raw.robokop.nodes@polars", "edges": "ingestion.raw.robokop.edges@polars"},
                outputs="fabricator.int.robokop.violations",
                name="validate_fabricated_robokop_datasets",
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
                func=nodes.validate_datasets,
                inputs={"nodes": "ingestion.raw.spoke.nodes@polars", "edges": "ingestion.raw.spoke.edges@polars"},
                outputs="fabricator.int.spoke.violations",
                name="validate_fabricated_spoke_datasets",
            ),
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.embiology"},
                outputs={
                    "nodes": "ingestion.raw.embiology.nodes@pandas",
                    "edges": "ingestion.raw.embiology.edges@pandas",
                },
                name="fabricate_embiology_datasets",
            ),
            node(
                func=_create_pairs,
                inputs=[
                    "ingestion.raw.drug_list",
                    "ingestion.raw.disease_list",
                    "params:fabricator.ground_truth.num_rows_per_category",
                ],
                outputs=[
                    "ingestion.raw.kgml_xdtd_ground_truth.positives",
                    "ingestion.raw.kgml_xdtd_ground_truth.negatives",
                ],
                name="create_gt_pairs",
            ),
            node(
                func=fabricate_datasets,
                inputs={
                    "fabrication_params": "params:fabricator.ec_gt",
                    "rtx_nodes": "ingestion.raw.rtx_kg2.nodes@pandas",
                },
                outputs={
                    "positive_edges": "ingestion.raw.ec_ground_truth.positives",
                    "negative_edges": "ingestion.raw.ec_ground_truth.negatives",
                },
                name="create_ec_gt_pairs",
            ),
            node(
                func=fabricate_datasets,
                inputs={
                    "fabrication_params": "params:fabricator.orchard",
                    "rtx_nodes": "ingestion.raw.rtx_kg2.nodes@pandas",
                },
                outputs={
                    "edges": "ingestion.raw.orchard.edges@pandas",
                },
                name="fabricate_orchard_datasets",
            ),
            node(
                func=generate_paths,
                inputs=[
                    "ingestion.raw.rtx_kg2.edges@pandas",
                    "ingestion.raw.kgml_xdtd_ground_truth.positives",
                    "ingestion.raw.kgml_xdtd_ground_truth.negatives",
                ],
                outputs="ingestion.raw.drugmech.edges@pandas",
                name="create_drugmech_pairs",
            ),
        ]
    )
