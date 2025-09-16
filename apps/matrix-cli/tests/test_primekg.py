"""Matrix CLI PrimeKG TestCase."""

import unittest

import polars as pl
from matrix_cli.io.primekg import coalesce_duplicate_columns, fix_curies, mondo_grouped_exploded


class PrimeKGTestCase(unittest.TestCase):
    """Suite of tests for PrimeKG functions."""

    def test_coalesce_duplicate_columns(self):
        """Tests coalescing columns upon a full join."""
        first = pl.DataFrame(
            {
                "name": ["qwer", "asdf", "zxcv"],
                "weight": [57.9, 72.5, None],
                "height": [1.56, 1.77, 1.65],
            }
        )

        second = pl.DataFrame(
            {
                "name": ["qwer", "asdf", "zxcv"],
                "weight": [None, None, 53.6],
            }
        )

        main = pl.DataFrame({"name": pl.Series([], dtype=pl.Utf8)})
        main = main.join(first, on=["name"], how="full", coalesce=True)
        main = main.join(second, on=["name"], how="full", coalesce=True)

        assert "weight_right" in main.columns
        main = coalesce_duplicate_columns(main, keep=["name"])
        assert "weight_right" not in main.columns

    def test_mondo_grouped_exploded(self):
        """Tests exploding MONDO groups."""
        df = pl.DataFrame(
            {
                "x_id": ["DB05271", "DB00492", "DB13956"],
                "x_source": ["DrugBank", "DrugBank", "DrugBank"],
                "y_id": ["1200_1134_15512_5080_100078", "1200_1134_15512_5080_100078", "1200_1134_15512_5080_100078"],
                "y_source": ["MONDO_grouped", "MONDO_grouped", "MONDO_grouped"],
            }
        )
        df = mondo_grouped_exploded(df.lazy()).collect()
        assert df.select(pl.len()).item() == 15

    def test_fix_curies(self):
        """Tests fixing CURIEs."""
        df = pl.DataFrame(
            {
                "x_id": ["9796", "7918", "2084", "5384"],
                "x_source": ["NCBI", "NCBI", "UBERON", "UBERON"],
                "y_id": ["56992", "9240", "105378952", "105378952"],
                "y_source": ["NCBI", "NCBI", "NCBI", "NCBI"],
                "subject": [None, None, None, None],
                "object": [None, None, None, None],
            }
        )

        df = fix_curies(df.lazy()).collect()
        assert df.get_column("subject").to_list() == [
            "NCBIGene:9796",
            "NCBIGene:7918",
            "UBERON:0002084",
            "UBERON:0005384",
        ]


if __name__ == "__main__":
    unittest.main()
