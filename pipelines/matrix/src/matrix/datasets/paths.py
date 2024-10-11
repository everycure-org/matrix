"""Module containing classes for representing and manipulating paths in a knowledge graph."""

import pandas as pd
from typing import Any, Dict, List

import neo4j
from .utils import BaseParquetDataset


class KGPaths:
    """Class representing a set of paths in a knowledge graph."""

    def __init__(self, num_hops: int = None, df: pd.DataFrame = None):
        """Initialize the SetPaths class.

        Args:
            num_hops: If provided, an empty dataset is initiated with the given number of hops.
            df: An optional pandas DataFrame containing the paths. Must conform to schema. Ignored if num_hops is provided.
        """
        if num_hops is not None:
            self.num_hops = num_hops
            self.df = pd.DataFrame(columns=self.get_columns(num_hops))
        elif df is not None:
            self.num_hops = self.get_num_hops(df)
            self.df = df
        else:
            raise ValueError("Either num_hops or df must be provided")

    @classmethod
    def get_columns(cls, num_hops: int) -> list[str]:
        """Return the list of columns in the dataframe for a given number of hops."""
        columns = ["source_name", "source_id", "source_type"]
        for i in range(1, num_hops):
            columns += [
                f"predicates_{i}",
                f"is_forward_{i}",
                f"intermediate_name_{i}",
                f"intermediate_id_{i}",
                f"intermediate_type_{i}",
            ]
        columns += [f"predicates_{num_hops}", f"is_forward_{num_hops}", "target_name", "target_id", "target_type"]
        return columns

    @classmethod
    def get_num_hops(cls, df: pd.DataFrame) -> int:
        """Return the number of hops in the paths represented by the dataframe.

        Raises:
            ValueError: If the dataframe does not conform to the expected schema.
        """
        num_hops = 0
        while set(cls.get_columns(num_hops + 1)).issubset(set(df.columns)):
            num_hops += 1

        if num_hops == 0:
            raise ValueError("df does not conform to the expected schema")
        return num_hops

    def __len__(self) -> int:
        """Return the number of paths in the set."""
        return len(self.df)

    def get_unique_pairs(self) -> pd.DataFrame:
        """Get the set of unique source and target nodes in the paths."""
        return self.df[["source_id", "target_id"]].drop_duplicates()

    def get_paths_for_pair(self, source_id: str, target_id: str) -> pd.DataFrame:
        """Get the paths for a given source and target node IDs."""
        return self.df[(self.df["source_id"] == source_id) & (self.df["target_id"] == target_id)]

    def add_paths_from_result(self, result: List[neo4j.graph.Path], extra_data: Dict[str, List[Any]] = None) -> None:
        """Add a path to the df from the results of a Neo4j query.

        Args:
            result: List of neo4j paths.
            extra_data: Dictionary of extra data to add to the dataframe.
        """
        if len(result) == 0:
            return

        # Initialize data dictionary
        if extra_data is None:
            data_dict = dict()
        else:
            data_dict = extra_data
        data_dict["source_name"] = []
        data_dict["source_id"] = []
        data_dict["source_type"] = []
        for i in range(1, self.num_hops):
            data_dict[f"predicates_{i}"] = []
            data_dict[f"is_forward_{i}"] = []
            data_dict[f"intermediate_name_{i}"] = []
            data_dict[f"intermediate_id_{i}"] = []
            data_dict[f"intermediate_type_{i}"] = []
        data_dict[f"predicates_{self.num_hops}"] = []
        data_dict[f"is_forward_{self.num_hops}"] = []
        data_dict["target_name"] = []
        data_dict["target_id"] = []
        data_dict["target_type"] = []

        # Collecting data
        for path in result:
            if len(path) != self.num_hops:
                raise ValueError(f"Path has {len(path)} hops, expected {self.num_hops}")

            nodes = [(node.get("name"), node.get("id"), node.get("category")) for node in path.nodes]
            edges_types = [type(edge).__name__ for edge in path.relationships]
            edge_directions = [edge.start_node == path.nodes[i] for i, edge in enumerate(path.relationships)]

            data_dict["source_name"].append(nodes[0][0])
            data_dict["source_id"].append(nodes[0][1])
            data_dict["source_type"].append(nodes[0][2])
            for i in range(1, self.num_hops):
                data_dict[f"predicates_{i}"].append(edges_types[i - 1])
                data_dict[f"is_forward_{i}"].append(edge_directions[i - 1])
                data_dict[f"intermediate_name_{i}"].append(nodes[i][0])
                data_dict[f"intermediate_id_{i}"].append(nodes[i][1])
                data_dict[f"intermediate_type_{i}"].append(nodes[i][2])
            data_dict[f"predicates_{self.num_hops}"].append(edges_types[-1])
            data_dict[f"is_forward_{self.num_hops}"].append(edge_directions[-1])
            data_dict["target_name"].append(nodes[-1][0])
            data_dict["target_id"].append(nodes[-1][1])
            data_dict["target_type"].append(nodes[-1][2])

        # Squash varying predicates into comma-separated strings
        full_new_data = pd.DataFrame(data_dict)
        predicate_cols = [f"predicates_{i}" for i in range(1, self.num_hops + 1)]
        non_predicate_cols = [col for col in full_new_data.columns if col not in predicate_cols]
        grouped = full_new_data.groupby(non_predicate_cols, as_index=False)
        agg_dict = {col: lambda x: ",".join(x.unique()) for col in predicate_cols}
        new_data = grouped.agg(agg_dict)
        new_data.reset_index(drop=True, inplace=True)

        # Add the new data to the existing dataframe
        self.df = pd.concat([self.df, new_data], ignore_index=True)


class KGPathsDataset(BaseParquetDataset):
    """Dataset adaptor to read KGPaths using Kedro's dataset functionality."""

    def _load(self) -> KGPaths:
        return self._load_with_retry(KGPaths, result_class_arg="df")


# # Define Kedro datasets for each hop count

# class TwoHopPaths(KGPaths):
#     """Class representing a set of two-hop paths in a knowledge graph."""

#     def __init__(self, df: pd.DataFrame = None):
#         super().__init__(num_hops=2, df=df)

# class TwoHopPathsDataset(BaseParquetDataset):
#     """Dataset adaptor to read TwoHopPaths using Kedro's dataset functionality."""

#     def _load(self) -> KGPaths:
#         return self._load_with_retry(TwoHopPaths)

# class ThreeHopPaths(KGPaths):
#     """Class representing a set of three-hop paths in a knowledge graph."""

#     def __init__(self, df: pd.DataFrame = None):
#         super().__init__(num_hops=3, df=df)

# class ThreeHopPathsDataset(BaseParquetDataset):
#     """Dataset adaptor to read ThreeHopPaths using Kedro's dataset functionality."""

#     def _load(self) -> KGPaths:
#         return self._load_with_retry(ThreeHopPaths)
#


# Alternative implementation using metaclasses
# def create_n_hop_paths_class(n: int) -> Type[KGPaths]:
#     """
#     Create a subclass of KGPaths for n-hop paths.

#     Args:
#         n: The number of hops.
#     """
#     class NHopPaths(KGPaths):
#         """Class representing a set of n-hop paths in a knowledge graph."""

#         def __init__(self, df: pd.DataFrame = None):
#             super().__init__(num_hops=n, df=df)

#         def __repr__(self):
#             return f"<{self.__class__.__name__} object>"

#    NHopPaths.__name__ = f"Paths{n}Hops"
#     return NHopPaths

# def create_n_hop_dataset_class(n: int) -> Type[BaseParquetDataset]:
#     """Create a dataset adaptor for n-hop paths."""
#     class NHopPathsDataset(BaseParquetDataset):
#         """Dataset adaptor for n-hop paths."""

#         def _load(self) -> KGPaths:
#             NHopPaths = create_n_hop_paths_class(n)
#             return self._load_with_retry(NHopPaths)

#     NHopPathsDataset.__name__ = f"PathsDataset{n}Hops"
#     return NHopPathsDataset

# TwoHopPathsDataset = create_n_hop_dataset_class(2)
# ThreeHopPathsDataset = create_n_hop_dataset_class(3)
# # Add more as needed
