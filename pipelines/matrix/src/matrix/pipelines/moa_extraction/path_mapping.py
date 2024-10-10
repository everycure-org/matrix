"""Module containing classes for mapping DrugMechDB paths to the knowledge graph."""

from typing import Dict, Any


from matrix.datasets.paths import KGPaths
from matrix.pipelines.moa_extraction.neo4j_runners import Neo4jRunner


class PathMapper:
    """Class for mapping DrugMechDB paths to the knowledge graph."""

    def run(self, runner: Neo4jRunner, drug_mech_db: Dict[str, Any], synonymizer_endpoint: str) -> KGPaths:
        """Run the path mapping.

        Args:
            runner: The Neo4j runner.
            drug_mech_db: The DrugMechDB indication paths.
            synonymizer_endpoint: The endpoint of the synonymizer.
        """
        paths = KGPaths(num_hops=2)
        return paths
