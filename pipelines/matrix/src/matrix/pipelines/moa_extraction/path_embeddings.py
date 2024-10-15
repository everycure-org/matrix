"""Module containing strategies for embedding knowledge graph paths."""

import pandas as pd
import abc
from typing import List

import torch
from torch import Tensor

from matrix.datasets.paths import KGPaths


class OneHotEncoder:
    """Class representing a PyTorch one-hot encoder for a set of strings."""

    def __init__(self, strings: List[str]):
        """Initialize the one-hot encoder.

        Args:
            strings: The list of strings to encode.
        """
        self.strings = strings
        self.one_hot = torch.eye(len(strings))

    def get_encoding(self, string: str) -> Tensor:
        """Get the one-hot encoding of a string.

        Args:
            string: The string to encode.
        """
        return self.one_hot[self.strings.index(string)]

    def get_sum_encoding(self, strings: List[str]) -> Tensor:
        """Get the sum of the one-hot encodings of a list of strings.

        Args:
            strings: The list of strings to encode.
        """
        return torch.sum(torch.stack([self.get_encoding(string) for string in strings]), dim=0)


class PathEmbeddingStrategy(abc.ABC):
    """Abstract class representing a path embedding strategy."""

    @abc.abstractmethod
    def generate_embeddings(self, paths_data: KGPaths) -> pd.DataFrame:
        """Generate embeddings for the paths data.

        Args:
            paths_data: The paths data.
        """
        ...


class TypesRelationsAndDirections(PathEmbeddingStrategy):
    """Class representing a path embedding strategy that uses node types, edge relations and edge directions."""

    def run(
        self, paths_data: KGPaths, category_encoder: OneHotEncoder, relation_encoder: OneHotEncoder
    ) -> pd.DataFrame:
        """Generate embeddings for the paths data.

        Args:
            paths_data: The paths data.
        """
        # num_hops = paths_data.num_hops
        # breakpoint()
        return ...


# TODO Types and relations embeddings
