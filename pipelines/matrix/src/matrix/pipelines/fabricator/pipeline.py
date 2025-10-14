import itertools
import random
from decimal import Decimal

import networkx as nx
import numpy as np
import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline
from matrix_fabricator.fabrication import fabricate_datasets

from matrix.utils.validation import validate


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


def _attach_scores(base_matrix: pd.DataFrame, skew: float, seed: int) -> pd.DataFrame:
    """Helper function to attach scores to fabricated run comparison matrices.

    Args:
        base_matrix: Base matrix of pairs to attach scores to.
        skew: Positive float, higher corresponds to better model
        seed: Seed for the random number generator.

    Returns:
        matrix with scores attached.
    """
    np.random.seed(seed)
    size = len(base_matrix)

    # Sample scores from Beta distributions
    pos_scores = pd.Series(np.random.beta(a=skew, b=1, size=size))  # Skewed to 1
    offlabel_scores = pd.Series(np.random.beta(a=skew / 2, b=1, size=size))  # Skewed to 1, but less than pos_scores
    neg_scores = pd.Series(np.random.beta(a=1, b=skew, size=size))  # Skewed to 0
    unk_scores = pd.Series(np.random.beta(a=2, b=2, size=size))  # Centered around 0.5

    # Attach scores to matrix
    matrix = base_matrix.copy(deep=True)
    matrix["treat score"] = pos_scores.where(
        pd.Series(matrix["is_known_positive"]),
        offlabel_scores.where(
            pd.Series(matrix["off_label"]), neg_scores.where(pd.Series(matrix["is_known_negative"]), unk_scores)
        ),
    )

    return matrix


def _add_test_set_columns(df: pd.DataFrame, test_set_proportion: float, test_set_name: str):
    """Add a Random boolean valued column to the dataframe."""
    df[test_set_name] = np.random.choice([True, False], size=len(df), p=[test_set_proportion, 1 - test_set_proportion])
    return df


def fabricate_run_comparison_matrices(
    N: int = 15, skew_good_model: float = 10, skew_bad_model: float = 2, test_set_proportion: float = 0.2
):
    """Generate fabricated matrix predictions data to test the run comparison pipeline."""
    np.random.seed(0)

    # Generate base matrices of drug-disease pairs
    drugs_df = pd.DataFrame({"source": [f"drug_{i}" for i in range(N)]})
    diseases_df = pd.DataFrame({"target": [f"disease_{i}" for i in range(N)]})
    base_matrix_fold_1 = pd.merge(drugs_df, diseases_df, how="cross")
    base_matrix_fold_1 = _add_test_set_columns(base_matrix_fold_1, test_set_proportion, "is_known_positive")
    base_matrix_fold_1 = _add_test_set_columns(base_matrix_fold_1, test_set_proportion, "is_known_negative")
    base_matrix_fold_1 = _add_test_set_columns(base_matrix_fold_1, test_set_proportion, "off_label")
    base_matrix_fold_2 = base_matrix_fold_1.copy(deep=True)
    base_matrix_fold_2 = _add_test_set_columns(base_matrix_fold_2, test_set_proportion, "is_known_positive")
    base_matrix_fold_2 = _add_test_set_columns(base_matrix_fold_2, test_set_proportion, "is_known_negative")
    base_matrix_fold_2 = _add_test_set_columns(base_matrix_fold_2, test_set_proportion, "off_label")

    # Generate scores and return
    return {
        "matrix_fold_1_good_model": _attach_scores(base_matrix_fold_1, skew_good_model, seed=1),
        "matrix_fold_2_good_model": _attach_scores(base_matrix_fold_2, skew_good_model, seed=2),
        "matrix_fold_1_bad_model": _attach_scores(base_matrix_fold_1, skew_bad_model, seed=3),
        "matrix_fold_2_bad_model": _attach_scores(base_matrix_fold_2, skew_bad_model, seed=4),
    }


def format_infores_catalog(fabrication_params: dict) -> dict:
    """Generate and format infores catalog YAML structure directly from params."""
    result = fabricate_datasets(fabrication_params=fabrication_params)
    information_resources = result["information_resources"]

    # Format the data (existing logic)
    resources_list = information_resources.to_dict("records")
    for resource in resources_list:
        for key, value in resource.items():
            if pd.isna(value):
                resource[key] = None
            elif isinstance(value, str) and "|" in value:
                resource[key] = value.split("|")
    return {"information_resources": resources_list}


def format_reusabledata_json(fabrication_params: dict) -> list:
    """Generate and format reusabledata JSON structure directly from params."""
    result = fabricate_datasets(fabrication_params=fabrication_params)
    data = result["data"]

    # Format the data (existing logic)
    resources_list = data.to_dict("records")
    for resource in resources_list:
        for key, value in resource.items():
            if pd.isna(value):
                resource[key] = None
            elif key == "id":
                resource[key] = str(value)
            elif isinstance(value, str) and "|" in value and "categories" in key:
                resource[key] = value.split("|")
            elif isinstance(value, Decimal):
                resource[key] = float(value)
    return resources_list


def format_kgregistry_yaml(fabrication_params: dict) -> dict:
    """Generate and format kg-registry YAML structure directly from params."""
    result = fabricate_datasets(fabrication_params=fabrication_params)
    resources = result["resources"]

    # Format the data (existing logic)
    resources_list = resources.to_dict("records")
    for resource in resources_list:
        for key, value in resource.items():
            if pd.isna(value):
                resource[key] = None
            elif isinstance(value, str) and "|" in value and "domains" in key:
                resource[key] = value.split("|")
    return {"resources": resources_list}


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
                func=validate,
                inputs={"nodes": "ingestion.raw.rtx_kg2.nodes@polars", "edges": "ingestion.raw.rtx_kg2.edges@polars"},
                outputs="fabricator.int.rtx_kg2.violations",
                name="validate_fabricated_kg2_datasets",
            ),
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.primekg"},
                outputs={
                    "nodes": "ingestion.raw.primekg.nodes@pandas",
                    "edges": "ingestion.raw.primekg.edges@pandas",
                },
                name="fabricate_primekg_datasets",
            ),
            node(
                func=validate,
                inputs={"nodes": "ingestion.raw.primekg.nodes@polars", "edges": "ingestion.raw.primekg.edges@polars"},
                outputs="fabricator.int.primekg.violations",
                name="validate_fabricated_primekg_datasets",
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
                func=validate,
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
                func=validate,
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
                    "fabrication_params": "params:fabricator.drugbank_gt",
                    "rtx_nodes": "ingestion.raw.rtx_kg2.nodes@pandas",
                },
                outputs={
                    "positive_edges": "ingestion.raw.drugbank_ground_truth.positives",
                    "negative_edges": "ingestion.raw.drugbank_ground_truth.negatives",
                },
                name="create_drugbank_gt_pairs",
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
            node(
                func=format_kgregistry_yaml,
                inputs={"fabrication_params": "params:fabricator.document_kg.kgregistry"},
                outputs="document_kg.raw.kgregistry",
                name="format_kgregistry",
            ),
            node(
                func=format_reusabledata_json,
                inputs={"fabrication_params": "params:fabricator.document_kg.reusabledata"},
                outputs="document_kg.raw.reusabledata",
                name="format_reusabledata",
            ),
            node(
                func=format_infores_catalog,
                inputs={"fabrication_params": "params:fabricator.document_kg.infores_catalog"},
                outputs="document_kg.raw.infores",
                name="format_infores_catalog",
            ),
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.document_kg.matrix_curated_pks"},
                outputs={"data": "document_kg.raw.matrix_curated_pks@pandas"},
                name="fabricate_matrix_curated_pks",
            ),
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.document_kg.matrix_reviews_pks"},
                outputs={"data": "document_kg.raw.matrix_reviews_pks@pandas"},
                name="fabricate_matrix_reviews_pks",
            ),
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.document_kg.mapping_kgregistry_infores"},
                outputs={"mappings": "document_kg.raw.mapping_kgregistry_infores"},
                name="fabricate_mapping_kgregistry_infores",
            ),
            node(
                func=fabricate_datasets,
                inputs={"fabrication_params": "params:fabricator.document_kg.mapping_reusabledata_infores"},
                outputs={"mappings": "document_kg.raw.mapping_reusabledata_infores"},
                name="fabricate_mapping_reusabledata_infores",
            ),
            node(
                func=fabricate_run_comparison_matrices,
                inputs=[
                    "params:fabricator.run_comparison.matrix_size",
                    "params:fabricator.run_comparison.skew_good_model",
                    "params:fabricator.run_comparison.skew_bad_model",
                    "params:fabricator.run_comparison.test_set_proportion",
                ],
                outputs={
                    "matrix_fold_1_good_model": "fabricator.raw.run_comparison_matrices.matrix_fold_1_good_model",
                    "matrix_fold_2_good_model": "fabricator.raw.run_comparison_matrices.matrix_fold_2_good_model",
                    "matrix_fold_1_bad_model": "fabricator.raw.run_comparison_matrices.matrix_fold_1_bad_model",
                    "matrix_fold_2_bad_model": "fabricator.raw.run_comparison_matrices.matrix_fold_2_bad_model",
                },
                name="fabricate_run_comparison_matrices",
            ),
        ]
    )
