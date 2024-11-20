"""Module containing classes for sampling negative paths."""

import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import abc

from matrix.datasets.paths import KGPaths
from matrix.pipelines.moa_extraction.neo4j_query_clauses import (
    generate_match_clause,
    generate_edge_omission_where_clause,
)
from matrix.pipelines.embeddings.nodes import GraphDB


class PathGenerator(abc.ABC):
    """Abstract class representing a KG paths generator."""

    @abc.abstractmethod
    def run(self, runner: GraphDB) -> KGPaths:
        """Generate the paths."""
        ...


class AllPathsWithRules(PathGenerator):
    """
    Generates all paths between the given source drug and target disease that match the given tag rules.
    """

    def __init__(
        self,
        edge_omission_rules: Dict[str, List[str]],
        num_hops: int,
        unidirectional: bool = True,
        num_limit: int = None,
    ):
        """Initialise the AllPathsWithRules.

        edge_omission_rules: The edge omission rules to match.
        num_hops: The number of hops in the paths.
        unidirectional: Whether to sample unidirectional paths only.
        num_limit: The maximum number of paths to return. If None, all paths are returned.
        """
        self.edge_omission_rules = edge_omission_rules
        self.num_hops = num_hops
        self.unidirectional = unidirectional
        self.num_limit = num_limit

    def run(self, runner: GraphDB, drug: str, disease: str) -> KGPaths:
        """Generate the paths.

        Args:
            runner: The GraphDB object representing the KG.
            drug: The source drug node ID.
            disease: The target disease node ID.
        """
        match_clause = generate_match_clause(num_hops=self.num_hops, unidirectional=self.unidirectional)
        where_clause = generate_edge_omission_where_clause(
            edge_omission_rules=self.edge_omission_rules, num_hops=self.num_hops
        )
        return_clause = KGPaths.generate_return_clause(limit=self.num_limit)
        query = f"""
        MATCH path = (start: %{{id:'{drug}'}}){match_clause}(end: %{{id:'{disease}'}})
        WHERE {where_clause}
        {return_clause}
        """
        result = runner.run(query)
        paths = KGPaths(num_hops=self.num_hops)
        paths.add_paths_from_result(result)
        return paths


class ReplacementPathSampler(PathGenerator):
    """Samples random paths between the source and target nodes in a given paths dataset."""

    def __init__(
        self,
        num_replacement_paths: int,
        edge_omission_rules: Dict[str, List[str]],
        unidirectional: bool = True,
        random_state: int = 1,
    ):
        """Initialise the ReplacementPathSampler.

        Args:
            num_replacement_paths: The number of replacement paths to sample for each positive path.
            edge_omission_rules: The edge omission rules to match.
            unidirectional: Whether to sample unidirectional paths only.
            random_state: The random state.
        """
        self.num_replacement_paths = num_replacement_paths
        self.unidirectional = unidirectional
        self.edge_omission_rules = edge_omission_rules
        self.random_state = random_state

    def run(self, paths: KGPaths, runner: GraphDB) -> KGPaths:
        """Sample random paths given a set of reference positive paths.

        self.num_replacement_paths random paths are sampled for each reference path, between the same source and target nodes.

        FUTURE: Create a subclass where this method is parallelised.

        paths: The reference paths dataset.
        runner: The GraphDB object representing the KG.
        """
        num_hops = paths.num_hops
        negative_paths = KGPaths(num_hops=num_hops)
        for _, row in tqdm(
            paths.get_unique_pairs().iterrows(), desc="Sampling negative paths with replacement strategy..."
        ):
            new_paths = self.run_single_pair(
                row["source_id"], row["target_id"], row["count"] * self.num_replacement_paths, runner, num_hops
            )
            negative_paths.df = pd.concat([negative_paths.df, new_paths.df])

        return negative_paths

    def run_single_pair(self, drug: str, disease: str, count: int, runner: GraphDB, num_hops: int) -> KGPaths:
        """Sample negative paths for the given source drug and target disease.

        drug: The drug node ID.
        disease: The disease node ID.
        count: The number of negative paths to sample.
        runner: The GraphDB object representing the KG.
        num_hops: The number of hops in the paths.
        """
        all_paths_generator = AllPathsWithRules(
            edge_omission_rules=self.edge_omission_rules, num_hops=num_hops, unidirectional=self.unidirectional
        )
        all_paths = all_paths_generator.run(runner, drug, disease)

        if len(all_paths.df) <= count:  # No need to sample if total number of paths is less than count
            return all_paths

        new_paths_df = all_paths.df.sample(n=count, random_state=self.random_state)
        return KGPaths(df=new_paths_df)
