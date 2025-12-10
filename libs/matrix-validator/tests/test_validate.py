"""Test Validation."""

import json
import os.path
import unittest
from importlib import resources as il_resources

import polars as pl
from biolink_model import prefixmaps

from matrix_validator.validator_polars import ValidatorPolarsDataFrameImpl, ValidatorPolarsFileImpl
from matrix_validator.validator_schema import ValidatorPanderaImpl


class TestValidate(unittest.TestCase):
    """Test validate."""

    def test_version_type(self):
        """Test validation method."""
        validator = ValidatorPanderaImpl(None, None)
        validator.validate(None)

    def test_biolink_prefix_class_mapping(self):
        """Test extraction of biolink preferred prefix per category mapping."""
        preferred_prefixes_per_class = json.loads(il_resources.files(prefixmaps).joinpath("preferred_prefixes_per_class.json").read_text())
        class_prefix_map = {
            item["class_name"]: [prefix["prefix"] for prefix in item["prefix_map"]]
            for item in preferred_prefixes_per_class["biolink_class_prefixes"]
        }

        print(class_prefix_map)

        prefix_class_map = {}

        for key in class_prefix_map.keys():
            prefixes = class_prefix_map[key]

            for prefix in prefixes:
                if prefix not in prefix_class_map:
                    prefix_class_map[prefix] = set()
                prefix_class_map[prefix].add(key)

        print(prefix_class_map)

    def test_validate_dataframe_rtxkg2(self):
        """Test validation method from DataFrame implementation."""
        test_nodes = os.path.join("tests/data/testdata_rtxkg2-kg_nodes.tsv")
        nodes_df = pl.scan_csv(test_nodes, separator="\t", has_header=True, ignore_errors=True).collect()
        test_edges = os.path.join("tests/data/testdata_rtxkg2-kg_edges.tsv")
        edges_df = pl.scan_csv(test_edges, separator="\t", has_header=True, ignore_errors=True).collect()
        validator = ValidatorPolarsDataFrameImpl(nodes=nodes_df, edges=edges_df)
        ret = validator.validate()
        print("\n".join(validator.violations))
        self.assertTrue(ret == 1)

    def test_validate_dataframe_robokop(self):
        """Test validation method from DataFrame implementation."""
        # robokop
        test_nodes = os.path.join("tests/data/testdata_robokop-kg_nodes.tsv")
        nodes_df = pl.scan_csv(test_nodes, separator="\t", has_header=True, ignore_errors=True).collect()
        test_edges = os.path.join("tests/data/testdata_robokop-kg_edges.tsv")
        edges_df = pl.scan_csv(test_edges, separator="\t", has_header=True, ignore_errors=True).collect()
        validator = ValidatorPolarsDataFrameImpl(nodes=nodes_df, edges=edges_df)
        ret = validator.validate()
        print("\n".join(validator.violations))
        self.assertTrue(ret == 1)

    def test_validate_file_rtxkg2(self):
        """Test validation method from DataFrame implementation."""
        # matrix_data_dir = ""
        # test_nodes = os.path.join(matrix_data_dir, "data/01_RAW/KGs/rtx_kg2/v2.10.0_validated.1/rtx-kg2_2.10.0_v2-2_nodes_100.tsv")
        # test_edges = os.path.join(matrix_data_dir, "data/01_RAW/KGs/rtx_kg2/v2.10.0_validated.1/rtx-kg2_2.10.0_v2-2_edges_100.tsv")
        test_nodes = os.path.join("tests/data/testdata_rtxkg2-kg_nodes.tsv")
        test_edges = os.path.join("tests/data/testdata_rtxkg2-kg_edges.tsv")
        validator = ValidatorPolarsFileImpl(nodes_file_path=test_nodes, edges_file_path=test_edges)

        self.assertTrue(validator.validate() == 1)

    def test_validate_file_robokop(self):
        """Test validation method from DataFrame implementation."""
        # matrix_data_dir = ""
        # test_nodes = os.path.join(matrix_data_dir, "data/test/raw/KGs/robokop-kg/30fd1bfc18cd5ccb/robokop-30fd1bfc18cd5ccb_nodes.tsv")
        # test_edges = os.path.join(matrix_data_dir, "data/test/raw/KGs/robokop-kg/30fd1bfc18cd5ccb/robokop-30fd1bfc18cd5ccb_edges.tsv")
        test_nodes = os.path.join("tests/data/testdata_robokop-kg_nodes.tsv")
        test_edges = os.path.join("tests/data/testdata_robokop-kg_edges.tsv")
        validator = ValidatorPolarsFileImpl(nodes_file_path=test_nodes, edges_file_path=test_edges)

        self.assertTrue(validator.validate() == 1)
