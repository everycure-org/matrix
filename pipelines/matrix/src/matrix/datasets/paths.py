"""Module containing classes for representing and manipulating paths in a knowledge graph."""

import pandas as pd


class KGPaths:
    """Class representing a set of paths in a knowledge graph."""

    def __init__(self, num_hops: int, paths_df: pd.DataFrame = None):
        """Initialize the SetPaths class.

        Args:
            num_hops: The number of hops in each path.
            paths_df: An optional pandas DataFrame containing the paths. Must conform to schema. Defaults to None.
        """
        self.num_hops = num_hops

        # Generate list of columns for paths_df
        self.columns = ["source_name", "source_id", "source_type"]
        for i in range(1, num_hops):
            self.columns += [
                f"predicates_{i}",
                f"intermediate_name_{i}",
                f"intermediate_id_{i}",
                f"intermediate_type_{i}",
            ]
        self.columns += [f"predicates_{num_hops}", "target_name", "target_id", "target_type"]

        # Initialize paths_df
        if paths_df is not None:
            self.paths_df = paths_df
        else:
            if paths_df.columns != self.columns:
                raise ValueError(f"paths_df must conform to schema {self.columns}")
            self.paths_df = paths_df

    def __len__(self) -> int:
        """Return the number of paths in the set."""
        return len(self.paths_df)

    def add_paths_from_result(self, paths) -> None:
        """Add a path to the paths_df from the results of a Neo4j query."""
        ...
