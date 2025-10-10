import polars as pl

# TODO type hints and comments


class MatrixPairs:
    def __init__(
        self,
        drugs_list: pl.DataFrame,
        diseases_list: pl.DataFrame,
        exclusion_pairs: pl.DataFrame,
        test_pairs: dict[str, pl.DataFrame],
    ):
        self.drugs_list = drugs_list
        self.diseases_list = diseases_list
        self.exclusion_pairs = exclusion_pairs
        self.test_pairs = test_pairs

    def to_lazyframe(self):
        matrix = (
            self.drugs_list.join(self.diseases_list, how="cross")
            .join(
                self.exclusion_pairs.with_columns(pl.lit(True).alias("is_excluded")),
                on=["source", "target"],
                how="left",
            )
            .filter(pl.col("is_excluded").is_null())
        )
        for col_name, test_pairs in self.test_pairs.items():
            matrix = matrix.join(
                test_pairs[col_name].with_columns(pl.lit(True).alias(col_name)), on=["source", "target"], how="left"
            ).fill_null(pl.lit(False))
        return pl.LazyFrame(matrix)

    @staticmethod
    def _pairs_equal_as_sets(df1: pl.DataFrame, df2: pl.DataFrame) -> bool:
        return (df1 == df1.join(df2, on=["source", "target"], how="inner")).all()

    def same_base_matrix(self, other: "MatrixPairs") -> bool:
        return self._pairs_equal_as_sets(self.drugs_list, other.drugs_list) and self._pairs_equal_as_sets(
            self.diseases_list, other.diseases_list
        )

    def __eq__(self, other: "MatrixPairs") -> bool:
        return (
            self.same_base_matrix(other)
            and self._pairs_equal_as_sets(self.exclusion_pairs, other.exclusion_pairs)
            and all(
                self._pairs_equal_as_sets(self.test_pairs[col_name], other.test_pairs[col_name])
                for col_name in self.test_pairs.keys()
            )
        )


def give_matrix_pairs_from_lazyframe(
    matrix: pl.LazyFrame,
    available_ground_truth_cols: list[str],
) -> MatrixPairs:
    return MatrixPairs(
        drugs_list=matrix["source"].unique(),
        diseases_list=matrix["target"].unique(),
        exclusion_pairs=matrix.join(matrix.select("source", "target"), on=["source", "target"], how="anti"),
        test_pairs={
            col_name: matrix.filter(col_name).select("source", "target") for col_name in available_ground_truth_cols
        },
    )


def harmonize_matrix_pairs(
    matrix_pairs: MatrixPairs,
    other: MatrixPairs,
) -> MatrixPairs:
    # Intersect drug and disease lists
    drugs_list = matrix_pairs.drugs_list.join(other.drugs_list, on="source", how="inner").unique()
    diseases_list = matrix_pairs.diseases_list.join(other.diseases_list, on="target", how="inner").unique()

    # Union exclusion pairs
    exclusion_pairs = pl.concat([matrix_pairs.exclusion_pairs, other.exclusion_pairs]).unique()

    # Intersect test pairs
    test_pairs = {
        col_name: matrix_pairs.test_pairs[col_name]
        .join(other.test_pairs[col_name], on=["source", "target"], how="inner")
        .unique()
        for col_name in matrix_pairs.test_pairs.keys()
    }

    return MatrixPairs(
        drugs_list=drugs_list, diseases_list=diseases_list, exclusion_pairs=exclusion_pairs, test_pairs=test_pairs
    )

    # init, materialize lazyframe, extract data (drugs, diseases, test pairs (dict), exclusion pairs)
    # to_lazyframe
    # _df_equal_as_sets (class method)
    # equals
    # same_base_matrix
    # harmonize
