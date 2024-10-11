"""Module containing classes for mapping DrugMechDB paths to the knowledge graph."""

from typing import Dict, Any, List
from tqdm import tqdm
import abc
import logging
import neo4j

from matrix.datasets.paths import KGPaths
from matrix.pipelines.moa_extraction.neo4j_runners import Neo4jRunner
from matrix.pipelines.preprocessing.nodes import resolve, normalize

logger = logging.getLogger(__name__)


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


def _parse_result(result: List[List[neo4j.graph.Path]]) -> List[neo4j.graph.Path]:
    """Parse the result of a Neo4j query into a list of paths.

    This is necessary because Neo4j returns a list of length 1 for each path.
    """
    return [path[0] for path in result]


class PathMapper(abc.ABC):
    """Abstract base class for mapping DrugMechDB paths to the knowledge graph."""

    @abc.abstractmethod
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

    def run(
        self,
        runner: Neo4jRunner,
        drug_mech_db: Dict[str, Any],
        synonymizer_endpoint: str,
        num_attempts: int = 5,
        max_entries: int = None,
    ) -> KGPaths:
        """Run the path mapping.

        Args:
            runner: The Neo4j runner.
            drug_mech_db: The DrugMechDB indication paths.
            synonymizer_endpoint: The endpoint of the synonymizer.
            num_attempts: The number of attempts to run the query.
            max_entries: The maximum number of entries to map. If None, all entries are mapped.
        """
        if max_entries:
            drug_mech_db = drug_mech_db[:max_entries]

        paths = KGPaths(self.num_hops)
        for db_entry in tqdm(drug_mech_db, desc="Mapping paths"):
            result = self.map_single_moa(runner, db_entry, synonymizer_endpoint, num_attempts)
            paths.add_paths_from_result(result)

        return paths

    def map_single_moa(
        self, runner: Neo4jRunner, db_entry: Dict[str, Any], synonymizer_endpoint: str, num_attempts: int = 5
    ) -> List[neo4j.graph.Path]:
        """Map a single DrugMechDB MOA.

        Args:
            runner: The Neo4j runner.
            db_entry: The DrugMechDB entry.
            synonymizer_endpoint: The endpoint of the synonymizer.
            num_attempts: The number of attempts to run the query.

        Returns:
            A list of the mapped paths. Empty list if no paths were found. None if the mapping failed.
        """

        def map_entity(name: str, curie: str) -> str:
            """Map a single entity with the synonymizer."""
            return _map_name_and_curie(name, curie, synonymizer_endpoint)

        # Get the drug and disease names and curies
        drug_name = db_entry["graph"]["drug"]
        drug_mesh_curie = db_entry["graph"]["drug_mesh"]
        drug_bank_curie = db_entry["graph"]["drugbank"]
        disease_name = db_entry["graph"]["disease"]
        disease_mesh_curie = db_entry["graph"]["disease_mesh"]

        # Map the drug and disease names and curies
        drug_mapped = map_entity(drug_name, drug_mesh_curie)
        if not drug_mapped:
            drug_mapped = map_entity(drug_bank_curie, drug_bank_curie)
        disease_mapped = map_entity(disease_name, disease_mesh_curie)

        if not drug_mapped or not disease_mapped:
            return None

        # Map intermediate nodes
        intermediate_db_entities = [node for node in db_entry["nodes"] if node["name"] not in [drug_name, disease_name]]
        mapped_int_ids = [map_entity(entity["name"], entity["id"]) for entity in intermediate_db_entities]
        mapped_int_ids = [entity for entity in mapped_int_ids if entity]  # Filter out Nones

        # Construct the match clause (e.g. "-[r1]->(a1)-[r2]->(a2)->[r3]->")
        edge_end = "->" if self.unidirectional else "-"
        match_clause_parts = [f"-[r1]{edge_end}"]
        for i in range(1, self.num_hops):
            match_clause_parts.append(f"(a{i})")
            match_clause_parts.append(f"-[r{i+1}]{edge_end}")
        match_clause = "".join(match_clause_parts)

        # Construct the where clause (e.g. "(a1.id in ['NCIT:C16325']) AND (a2.id in ['NCIT:C16325'])")
        where_clause = " AND ".join([f"(a{i}.id in {str(mapped_int_ids)})" for i in range(1, self.num_hops)])

        # Construct the neo4j query and run with several attempts (to account for connection issues)
        query = f"""MATCH p=(n:%{{id:'{drug_mapped}'}}){match_clause}(m:%{{id:'{disease_mapped}'}})
                    WHERE {where_clause}
                    RETURN DISTINCT p"""

        attempts = 0
        while attempts < num_attempts:
            try:
                result = runner.run(query)
                return _parse_result(result)
            except Exception as e:
                attempts += 1
                logger.warning(
                    f"Failed to map paths for {db_entry['graph']['_id']} (attempt {attempts}/{num_attempts}): {e}"
                )
        return None
