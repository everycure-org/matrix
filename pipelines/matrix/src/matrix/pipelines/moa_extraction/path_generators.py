"""Module containing classes for sampling negative paths."""

from typing import Dict, List
from tqdm import tqdm
import abc

from matrix.datasets.paths import KGPaths
from matrix.pipelines.moa_extraction.neo4j_runners import Neo4jRunner
from matrix.pipelines.moa_extraction.path_mapping import SetwisePathMapper


class PathGenerator(abc.ABC):
    """Abstract class representing a KG paths generator."""

    @abc.abstractmethod
    def run(self) -> KGPaths:
        """Generate the paths."""
        ...


class ReplacementPathSampler(PathGenerator):
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
        """Sample negative paths given a set of positive paths.

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
        """Sample negative paths for the given source drug and target disease.

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


class AllPathsWithTagRules(PathGenerator):
    """
    Generates all paths between the given source drug and target disease that match the given tag rules.
    """

    def __init__(
        self, tag_rules: Dict[str, List[str]], num_hops: int, unidirectional: bool = True, num_limit: int = None
    ):
        """Initialise the AllPathsWithTagRules.

        tag_rules: The tag rules to match. This takes the form of a dictionary with keys:
            'all', '1', '2', ...
            'all' are the edge tags to omit from all hops
            '1', '2', ... are edge tags to omit for the first hop, second hop, ... respectively.
            e.g. tag_rules = {'all': ['drug_disease'], '3': ['disease_disease']}
        num_hops: The number of hops in the paths.
        unidirectional: Whether to sample unidirectional paths only.
        num_limit: The maximum number of paths to return. If None, all paths are returned.
        """
        self.tag_rules = tag_rules
        self.num_hops = num_hops
        self.unidirectional = unidirectional
        self.num_limit = num_limit

    @classmethod
    def construct_where_clause(cls, tag_rules: dict, num_hops: int, prefix: str = "_moa_extraction_") -> str:
        """Construct the where clause for the query.

        E.g. NONE(r IN relationships(path) WHERE r._moa_extraction_drug_disease) AND (NOT r3._moa_extraction_disease_disease)

        Args:
            tag_rules: The tag rules to match.
            num_hops: The number of hops in the paths.
            prefix: The prefix for the tag.
        """
        where_clause_parts = []
        for tag in tag_rules["all"]:
            where_clause_parts.append(f"NONE(r IN relationships(path) WHERE r.{prefix}{tag})")
        for hop in range(1, num_hops + 1):
            if hop in tag_rules.keys():
                for tag in tag_rules[hop]:
                    where_clause_parts.append(f"NOT r{hop}.{prefix}{tag}")
        where_clause = " AND ".join(where_clause_parts)
        return where_clause

    def run(self, runner: Neo4jRunner, drug: str, disease: str) -> KGPaths:
        """Generate the paths.

        Args:
            runner: The Neo4j runner.
            drug: The source drug node ID.
            disease: The target disease node ID.
        """
        match_clause = SetwisePathMapper._construct_match_clause(
            num_hops=self.num_hops, unidirectional=self.unidirectional
        )
        where_clause = AllPathsWithTagRules.construct_where_clause(tag_rules=self.tag_rules, num_hops=self.num_hops)
        limit_clause = f"LIMIT {self.num_limit}" if self.num_limit is not None else ""
        query = f"""
        MATCH path = (start: %{{id:'{drug}'}}){match_clause}(end: %{{id:'{disease}'}})
        WHERE {where_clause}
        RETURN path {limit_clause}
        """
        result = runner.run(query)
        paths = KGPaths(num_hops=self.num_hops)
        paths.add_paths_from_result(result)
        return paths
