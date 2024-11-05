"""Module containing classes for mapping DrugMechDB paths to the knowledge graph."""

from typing import Dict, Any, List
from tqdm import tqdm
import abc
import logging
import random
import neo4j

from matrix.datasets.paths import KGPaths
from matrix.pipelines.moa_extraction.neo4j_query_clauses import return_clause
from matrix.pipelines.embeddings.nodes import GraphDB
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


class PathMapper(abc.ABC):
    """Abstract base class for mapping DrugMechDB paths to the knowledge graph."""

    @abc.abstractmethod
    def run(self, runner: GraphDB, drug_mech_db: Dict[str, Any], synonymizer_endpoint: str) -> KGPaths:
        """Run the path mapping.

        Args:
            runner: The GraphDB object representing the KG.
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

    def __init__(self, num_hops: int, unidirectional: bool, max_entries: int = None):
        """Initialize the SetwisePathMapper.

        Args:
            num_hops: The number of hops in the paths.
            unidirectional: Whether to map onto unidirectional paths only.
            max_entries: The maximum number of entries to map. If None, all entries are mapped.
        """
        self.num_hops = num_hops
        self.unidirectional = unidirectional
        self.max_entries = max_entries

    def run(
        self,
        runner: GraphDB,
        drug_mech_db: Dict[str, Any],
        synonymizer_endpoint: str,
        num_attempts: int = 5,
    ) -> KGPaths:
        """Run the path mapping.

        FUTURE: Create a subclass of SetwisePathMapper that paralellises this method.

        Args:
            runner: The GraphDB object representing the KG.
            drug_mech_db: The DrugMechDB indication paths.
            synonymizer_endpoint: The endpoint of the synonymizer.
            num_attempts: The number of attempts to run the query.
        """
        if self.max_entries:
            drug_mech_db = drug_mech_db[: self.max_entries]

        paths = KGPaths(num_hops=self.num_hops)
        for db_entry in tqdm(drug_mech_db, desc="Mapping paths"):
            result = self.map_single_moa(runner, db_entry, synonymizer_endpoint, num_attempts)
            drugmech_id = db_entry["graph"]["_id"]
            paths.add_paths_from_result(
                result, extra_data={"DrugMechDB_id": [drugmech_id] * len(result), "y": [1] * len(result)}
            )
        return paths

    @classmethod
    def _construct_match_clause(cls, num_hops: int, unidirectional: bool) -> str:
        """Construct an intermediate match clause.

        Example: "-[r1]->(a1)-[r2]->(a2)->[r3]->"

        num_hops: Number of hops in the path.
        unidirectional: Whether to map onto unidirectional paths only.
        """
        edge_end = "->" if unidirectional else "-"

        match_clause_parts = [f"-[r1]{edge_end}"]
        for i in range(1, num_hops):
            match_clause_parts.append(f"(a{i})")
            match_clause_parts.append(f"-[r{i+1}]{edge_end}")
        match_clause = "".join(match_clause_parts)

        return match_clause

    @classmethod
    def _construct_where_clause(cls, num_hops: int, intermediate_ids: List[str]) -> str:
        """Construct the where clause for a path mapping query.

        Example: "(a1.id in ['ID:1','ID:2']) AND (a2.id in ['ID:1','ID:2'])

        Args:
            num_hops: The number of hops in the path.
            intermediate_ids: The list of intermediate node IDs.


        """
        return " AND ".join([f"(a{i}.id in {str(intermediate_ids)})" for i in range(1, num_hops)])

    def map_single_moa(
        self, runner: GraphDB, db_entry: Dict[str, Any], synonymizer_endpoint: str, num_attempts: int = 5
    ) -> List[neo4j.graph.Path]:
        """Map a single DrugMechDB MOA.

        Args:
            runner: The GraphDB object representing the KG.
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

        # Construct the neo4j query
        match_clause = self._construct_match_clause(self.num_hops, self.unidirectional)
        where_clause = self._construct_where_clause(self.num_hops, mapped_int_ids)
        query = f"""MATCH p=(n:%{{id:'{drug_mapped}'}}){match_clause}(m:%{{id:'{disease_mapped}'}})
                    WHERE {where_clause}
                    {return_clause(limit=self.num_limit)}"""

        # Run the query with several attempts (to account for connection issues)
        attempts = 0
        while attempts < num_attempts:
            try:
                return runner.run(query)
            except Exception as e:
                attempts += 1
                logger.warning(
                    f"Failed to map paths for {db_entry['graph']['_id']} (attempt {attempts}/{num_attempts}): {e}"
                )
        return None


class TestPathMapper(PathMapper):
    """Path mapping strategy for the test environment.

    Simulates the path mapping by randomly sampling a set of paths in the test KG.
    """

    def __init__(self, num_hops: int, num_paths: int, unidirectional: bool, num_limit: int = None, *args, **kwargs):
        """Initialize the TestPathMapper.

        Args:
            num_hops: The number of hops in the paths.
            num_paths: The maximum number of paths in the KGPaths object.
            unidirectional: Whether to map onto unidirectional paths only.
            num_limit: The maximum number of paths to map. If None, all paths are returned.
            args: Additional ignored arguments.
            kwargs: Additional ignored keyword arguments.
        """
        self.num_hops = num_hops
        self.num_paths = num_paths
        self.unidirectional = unidirectional
        self.num_limit = num_limit

    def run(
        self,
        runner: GraphDB,
        drug_mech_db: Dict[str, Any],
        synonymizer_endpoint: str,
    ) -> KGPaths:
        """Run the path mapping.

        Args:
            runner: The GraphDB object representing the KG.
            drug_mech_db: The DrugMechDB indication paths.
            synonymizer_endpoint: The endpoint of the synonymizer.
        """
        match_clause = SetwisePathMapper._construct_match_clause(self.num_hops, self.unidirectional)
        query = f"""MATCH path=(n){match_clause}(m)
                    {return_clause(limit=self.num_limit)}"""
        result = runner.run(query)

        paths = KGPaths(num_hops=self.num_hops)
        random_ids = [random.randint(1, 1000) for _ in range(len(result))]
        paths.add_paths_from_result(result, extra_data={"DrugMechDB_id": random_ids, "y": [1] * len(result)})
        return paths
