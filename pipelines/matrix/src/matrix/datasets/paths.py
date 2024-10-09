"""Module containing classes for representing and manipulating paths in a knowledge graph."""

from kedro_datasets.pandas import ParquetDataset


class TwoHopPaths:
    """Class representing two-hop paths in a knowledge graph."""

    ...


class ThreeHopPaths:
    """Class representing three-hop paths in a knowledge graph."""

    ...


class TwoHopPathsDataset(ParquetDataset):
    """Dataset adaptor to read TwoHopPaths using Kedro's dataset functionality."""

    ...


class ThreeHopPathsDataset(ParquetDataset):
    """Dataset adaptor to read ThreeHopPaths using Kedro's dataset functionality."""

    ...
