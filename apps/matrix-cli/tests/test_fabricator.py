"""Matrix CLI Fabricator TestCase."""

import unittest

import polars as pl
from matrix_cli.io.fabricator import build_column_summary, create_edges_map, create_nodes_map, filtered_columns


class FabricatorTestCase(unittest.TestCase):
    """Suite of tests for Fabricator CLI functions."""

    def test_build_column_summary(self):
        """Build a summary of a DataFrame column."""
        first = pl.DataFrame(
            {
                "name": ["qwer", "asdf", "zxcv"],
                "weight": [57.9, 72.5, None],
                "height": [1.56, 1.77, 1.65],
            }
        )

        main = build_column_summary(first, "name")
        assert "name" in main.keys() and "datatype" in main.keys() and "samples" in main.keys()
        assert "asdf" in main["samples"]
        main = build_column_summary(first, "weight")
        assert 57.9 in main["samples"]

    def test_filtered_columns(self):
        """Determine columns after prefix exclusions."""
        first = pl.DataFrame(
            {
                "name": ["qwer", "asdf", "zxcv"],
                "weight_a": [57.9, 72.5, None],
                "weight_b": [57.9, 72.5, None],
                "height_a": [1.56, 1.77, 1.65],
                "height_b": [1.56, 1.77, 1.65],
            }
        )
        main = filtered_columns(first, ["weight", "height"])
        assert "height_a" not in main and "weight" not in main

    def test_create_nodes_map(self):
        """Create a nodes map."""
        first = pl.DataFrame(
            {
                "name": ["qwer", "asdf", "zxcv"],
                "weight": [57.9, 72.5, None],
                "height": [1.56, 1.77, 1.65],
            }
        )

        yaml = create_nodes_map(first, 3)
        assert "columns" in yaml.keys() and "name" in yaml["columns"] and "weight" in yaml["columns"]

    def test_create_edges_map(self):
        """Create an edges map."""
        first = pl.DataFrame(
            {
                "name": ["qwer", "asdf", "zxcv"],
                "weight": [57.9, 72.5, None],
                "height": [1.56, 1.77, 1.65],
            }
        )

        yaml = create_edges_map(first, 3)
        assert "columns" in yaml.keys() and "name" in yaml["columns"] and "weight" in yaml["columns"]


if __name__ == "__main__":
    unittest.main()
