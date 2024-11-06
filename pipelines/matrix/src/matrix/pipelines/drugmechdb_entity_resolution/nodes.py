"""Nodes for the DrugMechDB entity resolution pipeline."""

import pandas as pd
from typing import List
from tqdm import tqdm

from matrix.pipelines.preprocessing.nodes import resolve, normalize


def _map_name_and_curie(name: str, curie: str, endpoint: str) -> str:
    """Attempt to map an entity to the knowledge graph using either its name or curie.

    Args:
        name: The name of the node.
        curie: The curie of the node.
        endpoint: The endpoint of the synonymizer.

    Returns:
        The mapped curie. None if no mapping was found.
    """
    mapped_curie = normalize(curie, endpoint)
    if not mapped_curie:
        mapped_curie = normalize(name, endpoint)
    if not mapped_curie:
        mapped_curie = resolve(name, endpoint)

    return mapped_curie


def _map_several_ids_and_names(name: str, id_lst: List[str], synonymizer_endpoint: str) -> str:
    """Map a name and several IDs to the knowledge graph.

    Args:
        name: The name of the node.
        id_lst: List of IDs for the node.
        synonymizer_endpoint: The endpoint of the synonymizer.

    Returns:
        The mapped curie. None if no mapping was found.
    """
    id_lst = list(set(id_lst))  # Remove duplicates
    id_lst = [id for id in id_lst if id]  # Filter out Nones
    mapped_id = None
    for id in id_lst:
        mapped_id = _map_name_and_curie(name, id, synonymizer_endpoint)
        if mapped_id:
            break
    return mapped_id


def normalize_drugmechdb_entities(drug_mech_db: List[dict], synonymizer_endpoint: str) -> pd.DataFrame:
    """Normalize DrugMechDB entities.

    For drug and diseases nodes, there may be multiple IDs so we try to normalize with all of them to improve probability of mapping.

    Args:
        drug_mech_db: The DrugMechDB indication paths.
        synonymizer_endpoint: The endpoint of the synonymizer.
    """
    df = pd.DataFrame(columns=["DrugMechDB_ID", "DrugMechDB_name", "mapped_ID"])
    for entry in tqdm(drug_mech_db):
        for node in entry["nodes"]:
            mapped_id = None
            if node["name"] == entry["graph"]["drug"]:
                drugbank_id = entry["graph"]["drugbank"]
                drug_mesh_id = entry["graph"]["drug_mesh"]
                canonical_id = node["id"]
                mapped_id = _map_several_ids_and_names(
                    node["name"], [drugbank_id, drug_mesh_id, canonical_id], synonymizer_endpoint
                )
            elif node["name"] == entry["graph"]["disease"]:
                disease_mesh_id = entry["graph"]["disease_mesh"]
                canonical_id = node["id"]
                mapped_id = _map_several_ids_and_names(
                    node["name"], [disease_mesh_id, canonical_id], synonymizer_endpoint
                )
            else:
                mapped_id = _map_name_and_curie(node["name"], node["id"], synonymizer_endpoint)

            new_row = {"DrugMechDB_ID": node["id"], "DrugMechDB_name": node["name"], "mapped_ID": mapped_id}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.drop_duplicates(inplace=True)
    return df
