"""Module containing classes for sampling negative paths."""

import abc

from matrix.datasets.paths import KGPaths


class NegativePathSampler(abc.ABC):
    """Abstract class representing a negative path sampler."""

    @abc.abstractmethod
    def run(self) -> KGPaths:
        """Sample negative paths from the given paths."""
        ...


class MockNegativePathSampler(NegativePathSampler):  # TODO: REMOVE
    """Mock negative path sampler."""

    def run(self) -> KGPaths:
        """Sample negative paths from the given paths."""
        return None
