import networkx as nx
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
            columns=["source", "target", "drug|disease"],
        )

        # Remove duplicate pairs
        df = df.drop_duplicates()

        # Check that we still have enough fabricated pairs
        is_enough_generated = len(df) >= num or attempt > 100
        attempt += 1

    return df[:num], df[num : 2 * num]


def _create_ec_gt_pairs(positives: pd.DataFrame, negatives: pd.DataFrame):
    """Create ground truth pairs for EC data.
    Args:
        positives: DataFrame containing positive drug-disease pairs
        negatives: DataFrame containing negative drug-disease pairs
    Returns:
        DataFrame with formatted drug-disease pairs
    """
    # Rename columns consistently
    positives = positives.rename(columns={"source": "drug ID", "target": "disease ID"})
    negatives = negatives.rename(columns={"source": "drug ID", "target": "disease ID"})

    # Add indication flags
    positives["indication"] = True
    positives["contraindication"] = False
    negatives["indication"] = False
    negatives["contraindication"] = True

    # Combine positive and negative pairs
    df = pd.concat([positives, negatives])
    df = df.drop_duplicates()

    # Add required columns
    df["active ingredient"] = "dummy"
    df["drug label"] = df["drug ID"].copy().apply(lambda x: f"name_{x}")
    df["disease label"] = df["disease ID"].copy().apply(lambda x: f"name_{x}")
    df["disease name"] = df["disease label"]
    df["drug|disease"] = df["drug label"] + "|" + df["disease name"]

    return df


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
                func=_create_pairs,
                inputs=[
                    "ingestion.raw.drug_list",
                    "ingestion.raw.disease_list",
                ],
                outputs=[
                    "ingestion.raw.kgml_xdtd.positives",
                    "ingestion.raw.kgml_xdtd.negatives",
                ],
                name="create_gt_pairs",
            ),
            node(
                func=_create_ec_gt_pairs,
                inputs=[
                    "ingestion.raw.kgml_xdtd.positives",
                    "ingestion.raw.kgml_xdtd.negatives",
                ],
                outputs="ingestion.raw.ec_ground_truth.edges",
                name="create_ec_gt_pairs",
            ),
            node(
                func=generate_paths,
                inputs=[
                    "ingestion.raw.rtx_kg2.edges@pandas",
                    "ingestion.raw.kgml_xdtd.positives",
                    "ingestion.raw.kgml_xdtd.negatives",
                ],
                outputs="ingestion.raw.drugmech.edges@pandas",
                name="create_drugmech_pairs",
            ),
        ]
    )
