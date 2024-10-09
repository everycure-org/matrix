"""Module containing strategies for embedding knowledge graph paths."""

from typing import List

import torch
from torch import Tensor


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
