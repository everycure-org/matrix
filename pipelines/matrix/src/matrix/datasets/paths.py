"""Module containing classes for representing and manipulating paths in a knowledge graph."""

import pandas as pd

from .utils import BaseParquetDataset


class KGPaths:
    """Class representing a set of paths in a knowledge graph."""

    def __init__(self, num_hops: int, df: pd.DataFrame = None):
        """Initialize the SetPaths class.

        Args:
            num_hops: The number of hops in each path.
            df: An optional pandas DataFrame containing the paths. Must conform to schema. Defaults to None.
        """
        self.num_hops = num_hops

        # Generate list of columns for the dataframe
        self.columns = ["source_name", "source_id", "source_type"]
        for i in range(1, num_hops):
            self.columns += [
                f"predicates_{i}",
                f"is_forward_{i}",
                f"intermediate_name_{i}",
                f"intermediate_id_{i}",
                f"intermediate_type_{i}",
            ]
        self.columns += [f"predicates_{num_hops}", f"is_forward_{num_hops}", "target_name", "target_id", "target_type"]

        # Initialize df
        if df is not None:
            if df.columns != self.columns:
                raise ValueError(f"df must conform to schema {self.columns}")
            self.df = df
        else:
            self.df = pd.DataFrame(columns=self.columns)

    def __len__(self) -> int:
        """Return the number of paths in the set."""
        return len(self.df)

    def get_unique_pairs(self) -> pd.DataFrame:
        """Get the set of unique source and target nodes in the paths."""
        return self.df[["source_id", "target_id"]].drop_duplicates()

    def get_paths_for_pair(self, source_id: str, target_id: str) -> pd.DataFrame:
        """Get the paths for a given source and target node IDs."""
        return self.df[(self.df["source_id"] == source_id) & (self.df["target_id"] == target_id)]

    def add_paths_from_result(self, result) -> None:
        """Add a path to the df from the results of a Neo4j query."""
        ...


class TwoHopPaths(KGPaths):
    """Class representing a set of two-hop paths in a knowledge graph."""

    def __init__(self, df: pd.DataFrame = None):
        """Initialize the TwoHopPaths class.

        Args:
            df: An optional pandas DataFrame containing the paths. Must conform to schema. Defaults to None.
        """
        super().__init__(num_hops=2, df=df)


class ThreeHopPaths(KGPaths):
    """Class representing a set of three-hop paths in a knowledge graph."""

    def __init__(self, df: pd.DataFrame = None):
        """Initialize the ThreeHopPaths class.

        Args:
            df: An optional pandas DataFrame containing the paths. Must conform to schema. Defaults to None.
        """
        super().__init__(num_hops=3, df=df)


class TwoHopPathsDataset(BaseParquetDataset):
    """Dataset adaptor to read TwoHopPaths using Kedro's dataset functionality."""

    def _load(self) -> TwoHopPaths:
        return self._load_with_retry(TwoHopPaths)


class ThreeHopPathsDataset(BaseParquetDataset):
    """Dataset adaptor to read ThreeHopPaths using Kedro's dataset functionality."""

    def _load(self) -> ThreeHopPaths:
        return self._load_with_retry(ThreeHopPaths)


# Next: add to config and test
