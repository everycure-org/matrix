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
            bool_cols = [col for col in self.df.columns if "is_forward" in col]
            self.df[bool_cols] = self.df[bool_cols].astype(bool)
        elif df is not None:
            self.num_hops = self.get_num_hops(df)
            self.df = df
        else:
            raise ValueError("Either num_hops or df must be provided")

    @classmethod
    def get_columns(cls, num_hops: int) -> list[str]:
        """Return the list of columns in the dataframe for a given number of hops.

        Args:
            num_hops: The number of hops in the paths.
        """
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
        """Get the set of unique source and target nodes in the paths along with their counts."""
        unique_pairs = self.df[["source_id", "target_id"]].drop_duplicates()
        pair_counts = self.df.groupby(["source_id", "target_id"]).size().reset_index(name="count")
        unique_pairs = unique_pairs.merge(pair_counts, on=["source_id", "target_id"], how="left")
        return unique_pairs

    def get_paths_for_pair(self, source_id: str, target_id: str) -> pd.DataFrame:
        """Get the paths for a given source and target node IDs."""
        return self.df[(self.df["source_id"] == source_id) & (self.df["target_id"] == target_id)]

    @classmethod
    def _parse_result(cls, result: List[List[neo4j.graph.Path]]) -> List[neo4j.graph.Path]:
        """Parse the result of a Neo4j query into a list of paths.

        This is necessary because Neo4j returns a list of length 1 for each path.
        """
        return [path[0] for path in result]

    def add_paths_from_result(
        self, result: List[List[neo4j.graph.Path]], extra_data: Dict[str, List[Any]] = None
    ) -> None:
        """Add a path to the df from the results of a Neo4j query.

        The return clause of the query must be of the form:
            RETURN [n in nodes | n.name] as node_names,
                [n in nodes | n.id] as node_ids,
                [n in nodes | n.category] as node_categories,
                [r in rels | type(r)] as edge_types,
                [r in rels | startNode(r) = nodes[apoc.coll.indexOf(rels, r)]] as edge_directions

        Involves squashing multiple paths with the same nodes but different predicates into a single path.

        Args:
            result: Result of a Neo4j query (list of lists of length 1 of neo4j paths).
            extra_data: Dictionary of extra data to add to the dataframe.
        """
        if len(result) == 0:
            return

        # Initialize data dictionary
        if extra_data is None:
            data_dict = dict()
        else:
            data_dict = extra_data

        ## Generate dataframe with desired schema
        # Source (drug) node
        data_dict["source_name"] = [path["node_names"][0] for path in result]
        data_dict["source_id"] = [path["node_ids"][0] for path in result]
        data_dict["source_type"] = [path["node_categories"][0] for path in result]
        for i in range(1, self.num_hops):
            # All edges except the last one
            data_dict[f"predicates_{i}"] = [path["edge_types"][i - 1] for path in result]
            data_dict[f"is_forward_{i}"] = [path["edge_directions"][i - 1] for path in result]
            # All intermediate nodes
            data_dict[f"intermediate_name_{i}"] = [path["node_names"][i] for path in result]
            data_dict[f"intermediate_id_{i}"] = [path["node_ids"][i] for path in result]
            data_dict[f"intermediate_type_{i}"] = [path["node_categories"][i] for path in result]
        # Last edge
        data_dict[f"predicates_{self.num_hops}"] = [path["edge_types"][-1] for path in result]
        data_dict[f"is_forward_{self.num_hops}"] = [path["edge_directions"][-1] for path in result]
        # Target (disease) node
        data_dict["target_name"] = [path["node_names"][-1] for path in result]
        data_dict["target_id"] = [path["node_ids"][-1] for path in result]
        data_dict["target_type"] = [path["node_categories"][-1] for path in result]

        full_new_data = pd.DataFrame(data_dict)

        # Squash multiple predicates into comma-separated strings
        predicate_cols = [f"predicates_{i}" for i in range(1, self.num_hops + 1)]
        non_predicate_cols = [col for col in full_new_data.columns if col not in predicate_cols]
        grouped = full_new_data.groupby(non_predicate_cols, as_index=False)
        agg_dict = {col: lambda x: ",".join(x.unique()) for col in predicate_cols}
        new_data = grouped.agg(agg_dict).reset_index(drop=True)

        # Add the new data to the existing dataframe
        if len(self.df) > 0:
            self.df = pd.concat([self.df, new_data], ignore_index=True)
        else:
            self.df = new_data


class KGPathsDataset(BaseParquetDataset):
    """Dataset adaptor to read KGPaths using Kedro's dataset functionality."""

    def _load(self) -> KGPaths:
        return self._load_with_retry(KGPaths, result_class_arg="df")
