"""Module containing strategies for embedding knowledge graph paths."""

import pandas as pd
import numpy as np
import abc
from typing import List


from matrix.datasets.paths import KGPaths


class OneHotEncoder:
    """Class representing a PyTorch one-hot encoder for a set of strings."""

    def __init__(self, strings: List[str]):
        """Initialize the one-hot encoder.

        Args:
            strings: The list of strings to encode.
        """
        self.strings = strings
        self.one_hot = np.eye(len(strings))

    def get_encoding(self, string: str) -> np.ndarray:
        """Get the one-hot encoding of a string.

        Args:
            string: The string to encode.
        """
        return self.one_hot[self.strings.index(string)]

    def get_sum_encoding(self, strings: List[str]) -> np.ndarray:
        """Get the sum of the one-hot encodings of a list of strings.

        Args:
            strings: The list of strings to encode.
        """
        return np.sum(np.stack([self.get_encoding(string) for string in strings]), axis=0)


class PathEmbeddingStrategy(abc.ABC):
    """Abstract class representing a path embedding strategy."""

    @abc.abstractmethod
    def run(self, paths_data: KGPaths) -> np.array:
        """Generate embeddings for the paths data.

        Args:
            paths_data: The paths data.
        """
        ...


class TypesAndRelations(PathEmbeddingStrategy):
    """Class representing a path embedding strategy that uses node types and edge relations.

    Optionally, edge directions can be added as a feature."""

    def __init__(self, is_embed_directions: bool = False):
        """Initialize the path embedding strategy.

        Args:
            is_embed_directions: Whether to embed edge directions.
        """
        self.is_embed_directions = is_embed_directions

    def run(self, paths_data: KGPaths, category_encoder: OneHotEncoder, relation_encoder: OneHotEncoder) -> np.array:
        """Generate embeddings for the paths data.

        Args:
            paths_data: The paths data.
        """
        num_hops = paths_data.num_hops
        embeddings_lst = [
            self.run_single_embedding(row, category_encoder, relation_encoder, num_hops)
            for _, row in paths_data.df.iterrows()
        ]
        return np.array(embeddings_lst)

    def run_single_embedding(
        self, path: pd.Series, category_encoder: OneHotEncoder, relation_encoder: OneHotEncoder, num_hops: int
    ) -> np.array:
        """Generate embeddings for a single path.

        Args:
            path: Row from the paths data.
            category_encoder: The category encoder.
            relation_encoder: The relation encoder.
            num_hops: The number of hops.
        """
        source_type = category_encoder.get_encoding(path["source_type"])
        N_relations = len(relation_encoder.strings)
        first_vector = np.concatenate([source_type, np.zeros(N_relations)])
        if self.is_embed_directions:
            first_vector = np.concatenate([first_vector, np.zeros(1)])
        return np.ones(2, 2)
