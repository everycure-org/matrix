"""Module containing classes for sampling negative paths."""

from tqdm import tqdm
import abc

from matrix.datasets.paths import KGPaths
from matrix.pipelines.moa_extraction.neo4j_runners import Neo4jRunner
from matrix.pipelines.moa_extraction.path_mapping import SetwisePathMapper


class NegativePathSampler(abc.ABC):
    """Abstract class representing a negative path sampler."""

    @abc.abstractmethod
    def run(self) -> KGPaths:
        """Sample negative paths from the given paths."""
        ...


class ReplacementPathSampler(NegativePathSampler):
    """Samples random paths between the source and target nodes in a given paths dataset."""

    def __init__(self, num_replacement_paths: int, unidirectional: bool = True, random_state: int = 1):
        """Initialise the ReplacementPathSampler.

        Args:
            num_replacement_paths: The number of replacement paths to sample for each positive path.
            unidirectional: Whether to sample unidirectional paths only.
        """
        self.num_replacement_paths = num_replacement_paths
        self.unidirectional = unidirectional

    @classmethod
    def _construct_conditional_match_clause(cls, num_hops: int, unidirectional: bool) -> str:
        """Construct a conditional match clause.

        Example: "-[]-(b1: {id: nodes(path)[1].id})-[]-(b2: {id: nodes(path)[2].id})-[]-"

        num_hops: Number of hops in the path.
        unidirectional: Whether to map onto unidirectional paths only.
        """
        edge_end = "->" if unidirectional else "-"

        match_clause_parts = [f"-[]{edge_end}"]
        for i in range(1, num_hops):
            match_clause_parts.append(f"(b{i}: %{{id: nodes(path)[{i}].id}})")
            match_clause_parts.append(f"-[]{edge_end}")
        match_clause = "".join(match_clause_parts)

        return match_clause

    def run(self, paths: KGPaths, runner: Neo4jRunner) -> KGPaths:
        """Sample negative paths from the given paths.

        FUTURE: Create a subclass where this method is parallelised.

        paths: The reference paths dataset.
        runner: The Neo4j runner.
        """
        num_hops = paths.num_hops
        negative_paths = KGPaths(num_hops=num_hops)
        for _, row in tqdm(
            paths.get_unique_pairs().iterrows(), desc="Sampling negative paths with replacement strategy..."
        ):
            result = self.run_single_pair(row["source_id"], row["target_id"], row["count"], runner, num_hops)
            negative_paths.add_paths_from_result(result)

        return negative_paths

    def run_single_pair(self, drug: str, disease: str, count: int, runner: Neo4jRunner, num_hops: int) -> KGPaths:
        """Sample negative paths from the given paths.

        drug: The drug node ID.
        disease: The disease node ID.
        count: The number of negative paths to sample.
        runner: The Neo4j runner.
        num_hops: The number of hops in the paths.
        """
        basic_match_clause = SetwisePathMapper._construct_match_clause(
            num_hops=num_hops, unidirectional=self.unidirectional
        )
        conditional_match_clause = ReplacementPathSampler._construct_conditional_match_clause(
            num_hops=num_hops, unidirectional=self.unidirectional
        )

        query = f"""
        // Sample simple paths (one predicate per edge)
        MATCH path = (start: %{{id:'{drug}'}}){basic_match_clause}(end: %{{id:'{disease}'}})
        WHERE NONE(r IN relationships(path) WHERE r._moa_extraction_drug_disease)
        WITH path, rand() AS I
        ORDER BY I
        LIMIT {count}
        // Collect all possible predicates
        MATCH all_paths =(start: %{{id:'{drug}'}}){conditional_match_clause}(end: %{{id:'{disease}'}})
        return all_paths
        """

        return runner.run(query)
