"""Matrix CLI PrimeKG TestCase."""

import unittest

import polars as pl
from matrix_io_utils.robokop import (
    robokop_convert_boolean_columns_to_label_columns,
    robokop_strip_type_from_column_names,
)
from polars.testing import assert_frame_equal


class RobokopKGTestCase(unittest.TestCase):
    """Suite of tests for Robokop functions."""

    def test_robokop_convert_boolean_columns_to_label_columns(self):
        """Tests coalescing columns upon a full join."""
        orig_df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "category": ["biolink:Gene", "biolink:Gene", "biolink:Gene"],
                "MONDO_SUPERCLASS_qwer": ["true", None, None],
                "MONDO_SUPERCLASS_asdf": ["true", "true", None],
                "CHEBI_ROLE_qwer": [None, "true", None],
                "CHEBI_ROLE_asdf": [None, None, "true"],
            },
            infer_schema_length=0,
        )

        test_df = robokop_convert_boolean_columns_to_label_columns(orig_df.lazy())

        expected_df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "category": ["biolink:Gene", "biolink:Gene", "biolink:Gene"],
                "MONDO_SUPERCLASS": ["qwer|asdf", "asdf", ""],
                "CHEBI_ROLE": ["", "qwer", "asdf"],
            }
        )

        assert_frame_equal(test_df, expected_df)

    def test_robokop_strip_type_from_column_names(self):
        """Tests exploding MONDO groups."""
        orig_df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "category": ["biolink:Gene", "biolink:Gene", "biolink:Gene"],
                "qwer:boolean": [True, None, None],
                "asdf:int": [1, 2, 3],
                "zxcv:string[]": ["a|b|c", "a|b|c", "a|b|c"],
            },
            infer_schema_length=0,
        )

        test_df = robokop_strip_type_from_column_names(orig_df.lazy())

        expected_df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "category": ["biolink:Gene", "biolink:Gene", "biolink:Gene"],
                "qwer": [True, None, None],
                "asdf": [1, 2, 3],
                "zxcv": ["a|b|c", "a|b|c", "a|b|c"],
            }
        )

        assert_frame_equal(test_df, expected_df)


if __name__ == "__main__":
    unittest.main()
