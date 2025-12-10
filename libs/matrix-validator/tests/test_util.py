"""Test utility functions."""

import json
import unittest
from unittest import mock

import polars as pl
import tomllib

from matrix_validator.util import analyze_edge_types, get_valid_edge_types


class TestUtilFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_get_valid_edge_types(self):
        """Test get_valid_edge_types function."""
        # Call the function
        edge_types = get_valid_edge_types()

        # Assert it returns a list
        self.assertIsInstance(edge_types, list)

        # Assert the list contains dictionaries
        self.assertTrue(all(isinstance(item, dict) for item in edge_types))

        # Assert the list is not empty
        self.assertGreater(len(edge_types), 0)

        # Verify expected keys in each dictionary
        expected_keys = {"subject_category", "predicate", "object_category"}
        for item in edge_types:
            self.assertEqual(set(item.keys()), expected_keys)

        # Assert all values are strings
        for item in edge_types:
            self.assertTrue(all(isinstance(value, str) for value in item.values()))

        # Assert all values are prefixed with "biolink:"
        for item in edge_types:
            for value in item.values():
                self.assertTrue(value.startswith("biolink:"), f"Value '{value}' does not start with 'biolink:'")

        # Check unique values for each field to ensure proper variety
        subject_categories = {item["subject_category"] for item in edge_types}
        predicates = {item["predicate"] for item in edge_types}
        object_categories = {item["object_category"] for item in edge_types}

        # Verify we have multiple unique values for each field
        self.assertGreater(len(subject_categories), 1)
        self.assertGreater(len(predicates), 1)
        self.assertGreater(len(object_categories), 1)

    @mock.patch("matrix_validator.util.il_resources.files")
    def test_get_valid_edge_types_parsing(self, mock_files):
        """Test parsing of TSV data in get_valid_edge_types function."""
        # Create a mock for the file object
        mock_file = mock.MagicMock()
        mock_file.open.return_value.__enter__.return_value.read.return_value = (
            "subject_category\tpredicate\tobject_category\n"
            "biolink:Gene\tbiolink:related_to\tbiolink:Disease\n"
            "biolink:Gene\tbiolink:interacts_with\tbiolink:Protein\n"
            "biolink:Chemical\tbiolink:treats\tbiolink:Disease\n"
        )

        # Set up the mock for files().joinpath()
        mock_files.return_value.joinpath.return_value = mock_file

        # Call the function with mocked data
        edge_types = get_valid_edge_types()

        # Assert we get 3 items
        self.assertEqual(len(edge_types), 3)

        # Assert the expected values are present
        expected_items = [
            {"subject_category": "biolink:Gene", "predicate": "biolink:related_to", "object_category": "biolink:Disease"},
            {"subject_category": "biolink:Gene", "predicate": "biolink:interacts_with", "object_category": "biolink:Protein"},
            {"subject_category": "biolink:Chemical", "predicate": "biolink:treats", "object_category": "biolink:Disease"},
        ]

        for expected_item in expected_items:
            self.assertIn(expected_item, edge_types)

    @mock.patch("matrix_validator.util.il_resources.files")
    def test_get_valid_edge_types_malformed_line(self, mock_files):
        """Test handling of malformed lines in get_valid_edge_types."""
        # Create a mock for the file object
        mock_file = mock.MagicMock()
        mock_file.open.return_value.__enter__.return_value.read.return_value = (
            "subject_category\tpredicate\tobject_category\n"
            "biolink:Gene\tbiolink:related_to\tbiolink:Disease\n"
            "biolink:Gene\tbiolink:interacts_with\n"  # Missing object_category
            "biolink:Chemical\tbiolink:treats\tbiolink:Disease\n"
        )

        # Set up the mock for files().joinpath()
        mock_files.return_value.joinpath.return_value = mock_file

        # Call the function
        edge_types = get_valid_edge_types()

        # Should only return 2 valid items (skipping the malformed one)
        self.assertEqual(len(edge_types), 2)

        # The malformed item should not be included
        self.assertNotIn({"subject_category": "biolink:Gene", "predicate": "biolink:interacts_with"}, edge_types)

    def test_analyze_edge_types(self):
        """Test analyze_edge_types function."""
        # Create sample node data
        nodes_data = {
            "id": ["node1", "node2", "node3", "node4"],
            "category": ["biolink:Gene", "biolink:Disease", "biolink:Protein", "biolink:ChemicalSubstance"],
        }
        nodes_df = pl.DataFrame(nodes_data)

        # Create sample edge data
        edges_data = {
            "subject": ["node1", "node1", "node3", "node4", "node4", "node5"],
            "predicate": [
                "biolink:related_to",
                "biolink:affects",
                "biolink:interacts_with",
                "biolink:treats",
                "biolink:related_to",
                "biolink:related_to",
            ],
            "object": ["node2", "node3", "node1", "node2", "node3", "node2"],
        }
        edges_df = pl.DataFrame(edges_data)

        # Mock the get_valid_edge_types function to return a controlled set of valid edges
        with mock.patch("matrix_validator.util.get_valid_edge_types") as mock_get_valid_edge_types:
            mock_get_valid_edge_types.return_value = [
                {"subject_category": "biolink:Gene", "predicate": "biolink:related_to", "object_category": "biolink:Disease"},
                {"subject_category": "biolink:Protein", "predicate": "biolink:interacts_with", "object_category": "biolink:Gene"},
                {"subject_category": "biolink:ChemicalSubstance", "predicate": "biolink:treats", "object_category": "biolink:Disease"},
            ]

            # Call the function
            result = analyze_edge_types(nodes_df, edges_df)

            # Verify the result
            self.assertEqual(len(result), 6)  # 6 distinct edge types

            # Check counts
            counts_dict = {(row["subject_category"], row["predicate"], row["object_category"]): row["count"] for row in result.to_dicts()}

            # Valid edge types with their expected counts
            self.assertEqual(counts_dict[("biolink:Gene", "biolink:related_to", "biolink:Disease")], 1)
            self.assertEqual(counts_dict[("biolink:Protein", "biolink:interacts_with", "biolink:Gene")], 1)
            self.assertEqual(counts_dict[("biolink:ChemicalSubstance", "biolink:treats", "biolink:Disease")], 1)

            # Check validity
            validity_dict = {(row["subject_category"], row["predicate"], row["object_category"]): row["valid"] for row in result.to_dicts()}

            # These should be valid
            self.assertTrue(validity_dict[("biolink:Gene", "biolink:related_to", "biolink:Disease")])
            self.assertTrue(validity_dict[("biolink:Protein", "biolink:interacts_with", "biolink:Gene")])
            self.assertTrue(validity_dict[("biolink:ChemicalSubstance", "biolink:treats", "biolink:Disease")])

            # These should be invalid
            self.assertFalse(validity_dict[("biolink:Gene", "biolink:affects", "biolink:Protein")])
            self.assertFalse(validity_dict[("biolink:ChemicalSubstance", "biolink:related_to", "biolink:Protein")])

            # Check that we have a row for edges referencing unknown nodes
            unknown_edges = [
                row for row in result.to_dicts() if row["subject_category"] == "unknown" or row["object_category"] == "unknown"
            ]
            self.assertEqual(len(unknown_edges), 1)

    def test_analyze_edge_types_input_validation(self):
        """Test input validation in analyze_edge_types function."""
        # Create sample node data missing required columns
        invalid_nodes_df = pl.DataFrame(
            {
                "id": ["node1", "node2"],
                # Missing 'category' column
            }
        )

        # Create sample edge data missing required columns
        invalid_edges_df = pl.DataFrame(
            {
                "subject": ["node1", "node2"],
                # Missing 'predicate' and 'object' columns
            }
        )

        # Valid dataframes for comparison
        valid_nodes_df = pl.DataFrame({"id": ["node1", "node2"], "category": ["biolink:Gene", "biolink:Disease"]})

        valid_edges_df = pl.DataFrame(
            {"subject": ["node1", "node2"], "predicate": ["biolink:related_to", "biolink:affects"], "object": ["node2", "node1"]}
        )

        # Test missing column in nodes dataframe
        with self.assertRaises(ValueError) as context:
            analyze_edge_types(invalid_nodes_df, valid_edges_df)
        self.assertTrue("Required column 'category' not found" in str(context.exception))

        # Test missing columns in edges dataframe
        with self.assertRaises(ValueError) as context:
            analyze_edge_types(valid_nodes_df, invalid_edges_df)
        self.assertTrue("Required column" in str(context.exception))

    def test_config_toml(self):
        """Test toml config."""
        with open("config.toml", "rb") as config_file:
            config_contents = tomllib.load(config_file)

            if config_contents["biolink"]["supplemental_prefixes"]:
                supplemental_prefixes = config_contents["biolink"]["supplemental_prefixes"]
                for prefix in supplemental_prefixes:
                    print(prefix)

            for check in config_contents["edges_attribute_checks"]["checks"]:
                if "range" in check:
                    print(check["range"]["column"])

            for check in config_contents["nodes_attribute_checks"]["checks"]:
                if "range" in check:
                    print(check["range"]["column"])

    def test_scratch(self):
        """Test toml config."""

        def serialize_sets(obj):
            if isinstance(obj, set):
                return list(obj)

            return obj

        violations = {
            "The prefixes 'NCBITaxon,NCBIGene' are not within the Biolink preferred category mapping of 'biolink:Protein'",
            "The prefixes 'NCBIGene,UniProtKB' are not within the Biolink preferred category mapping of 'biolink:OrganismTaxon'",
            "The prefixes 'NCBITaxon,UniProtKB' are not within the Biolink preferred category mapping of 'biolink:Gene'",
        }
        violation_summary = {"check": "node_id_and_category_with_biolink_preferred_prefixes", "violations": violations}

        # values = set(["NCBITaxon", "UniProtKB"])
        print(json.dumps(violation_summary, default=serialize_sets))
