"""Module containing classes for mapping DrugMechDB paths to the knowledge graph."""

from typing import Dict, Any
from tqdm import tqdm
from abc import ABC, abstractmethod

from matrix.datasets.paths import KGPaths
from matrix.pipelines.moa_extraction.neo4j_runners import Neo4jRunner
from matrix.pipelines.preprocessing.nodes import resolve


def _map_name_and_curie(name: str, curie: str, endpoint: str) -> str:
    """Attempt to map an entity to the knowledge graph using either its name or curie.

    Args:
        name: The name of the node.
        curie: The curie of the node.
        endpoint: The endpoint of the synonymizer.
    """
    mapped_curie = resolve(name, endpoint)
    # if not mapped_curie:
    #     mapped_curie = normalize(curie, endpoint)

    return mapped_curie


class PathMapper(ABC):
    """Abstract base class for mapping DrugMechDB paths to the knowledge graph."""

    @abstractmethod
    def run(self, runner: Neo4jRunner, drug_mech_db: Dict[str, Any], synonymizer_endpoint: str) -> KGPaths:
        """Run the path mapping.

        Args:
            runner: The Neo4j runner.
            drug_mech_db: The DrugMechDB indication paths.
            synonymizer_endpoint: The endpoint of the synonymizer.

        Returns:
            KGPaths: The mapped paths.
        """
        pass


class SetwisePathMapper(PathMapper):
    """Class for mapping DrugMechDB paths to the knowledge graph.

    The strategy employed is as follows:or each drug-disease pair in the DrugMechDB, find all paths in the KG between them whose intermediate nodes also appear as intermediate nodes in the corresponding DrugMechDB MOA graph.

    """

    def __init__(self, num_hops: int, unidirectional: bool):
        """Initialize the SetwisePathMapper.

        Args:
            num_hops: The number of hops in the paths.
            unidirectional: Whether to map onto unidirectional paths only.
        """
        self.num_hops = num_hops
        self.unidirectional = unidirectional

    def run(self, runner: Neo4jRunner, drug_mech_db: Dict[str, Any], synonymizer_endpoint: str) -> KGPaths:
        """Run the path mapping.

        Args:
            runner: The Neo4j runner.
            drug_mech_db: The DrugMechDB indication paths.
            synonymizer_endpoint: The endpoint of the synonymizer.
        """
        drug_mech_db = drug_mech_db[:10]

        paths = KGPaths(self.num_hops)
        for dmdb_entry in tqdm(drug_mech_db, desc="Mapping paths"):
            result = self.map_single_moa(runner, dmdb_entry, synonymizer_endpoint)
            paths.add_paths_from_result(result)

        return paths

    def map_single_moa(self, runner: Neo4jRunner, dmdb_entry: Dict[str, Any], synonymizer_endpoint: str) -> KGPaths:
        """Map a single DrugMechDB MOA.

        Args:
            runner: The Neo4j runner.
            dmdb_entry: The DrugMechDB entry.
            synonymizer_endpoint: The endpoint of the synonymizer.
        """
        # Get the drug and disease names and curies
        drug_name = dmdb_entry["graph"]["drug"]
        drug_mesh_curie = dmdb_entry["graph"]["drug_mesh"]
        drug_bank_curie = dmdb_entry["graph"]["drugbank"]
        disease_name = dmdb_entry["graph"]["disease"]
        disease_mesh_curie = dmdb_entry["graph"]["disease_mesh"]

        # Map the drug and disease names and curies
        drug_mapped = _map_name_and_curie(drug_name, drug_mesh_curie, synonymizer_endpoint)
        if not drug_mapped:
            drug_mapped = _map_name_and_curie(drug_bank_curie, drug_bank_curie, synonymizer_endpoint)
        disease_mapped = _map_name_and_curie(disease_name, disease_mesh_curie, synonymizer_endpoint)

        if not drug_mapped or not disease_mapped:
            return None

        # Get intermediate nodes
