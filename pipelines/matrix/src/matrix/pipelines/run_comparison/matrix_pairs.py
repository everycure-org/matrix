import polars as pl
from func_tools import reduce


class MatrixPairs:
    """Sparse representation of a matrix drug-disease pairs dataframe, allowing for efficient computation of common elements for evaluation.

    Explanation:
        A set of matrix pairs may be expressed as:
            drugs_list x diseases_list - exclusion pairs
        where:
            - "x" denotes the cartesian product (i.e. cross join)
            - "-" denote the set difference (i.e. anti join)
        In addition, each type of test pairs is represented as its own dataframe.

        A matrix pairs dataframe, with "source" and "target" columns along with a boolean valued column for each type of test pairs,
        typically contains tens of millions of rows. Comparatively, this representation only requires a few tens of thousands of rows.
    """

    def __init__(
        self,
        drugs_list: pl.DataFrame,
        diseases_list: pl.DataFrame,
        exclusion_pairs: pl.DataFrame,
        test_pairs: dict[str, pl.DataFrame],
    ):
        """Initialize the MatrixPairs object.

        Args:
            drugs_list: Drugs list dataframe.
            diseases_list: Diseases list dataframe.
            exclusion_pairs: Exclusion pairs dataframe.
            test_pairs: Dictionary containing the dataframe of test pairs for each type of ground truth.
        """
        self.drugs_list = drugs_list
        self.diseases_list = diseases_list
        self.exclusion_pairs = exclusion_pairs
        self.test_pairs = test_pairs

    def to_lazyframe(self) -> pl.LazyFrame:
        """Convert the MatrixPairs object to a Polars LazyFrame.

        Returns:
            Polars LazyFrame representing the matrix pairs.
            Columns: "source", "target", and a boolean valued column for each type of test pairs.
        """
        # Generate pairs dataframe
        matrix = (
            pl.LazyFrame(self.drugs_list)
            .join(pl.LazyFrame(self.diseases_list), on=["source", "target"], how="cross")
            .join(
                pl.LazyFrame(self.exclusion_pairs).with_columns(pl.lit(True).alias("is_excluded")),
                on=["source", "target"],
                how="left",
            )
            .filter(pl.col("is_excluded").is_null())
            .drop("is_excluded")
        )

        # Join test pairs
        for col_name, test_pairs in self.test_pairs.items():
            matrix = matrix.join(
                pl.LazyFrame(test_pairs[col_name]).with_columns(pl.lit(True).alias(col_name)),
                on=["source", "target"],
                how="left",
            ).fill_null(pl.lit(False))

        return matrix

    @staticmethod
    def _pairs_equal_as_sets(df1: pl.DataFrame, df2: pl.DataFrame) -> bool:
        """Check if two pairs dataframes are equal as sets."""
        return (df1 == df1.join(df2, on=["source", "target"], how="inner")).all()

    def same_base_matrix(self, other: "MatrixPairs") -> bool:
        """Check if two MatrixPairs objects have the same base matrix, i.e the same drugs and diseases lists."""
        return self._pairs_equal_as_sets(self.drugs_list, other.drugs_list) and self._pairs_equal_as_sets(
            self.diseases_list, other.diseases_list
        )

    def __eq__(self, other: "MatrixPairs") -> bool:
        """Check if two MatrixPairs objects are equal.

        More precisely, check they represent the same set of drug-disease pairs and with the same test sets.
        """
        return (
            self.same_base_matrix(other)
            and self._pairs_equal_as_sets(self.exclusion_pairs, other.exclusion_pairs)
            and list(self.test_pairs.keys()) == list(other.test_pairs.keys())
            and all(
                self._pairs_equal_as_sets(self.test_pairs[col_name], other.test_pairs[col_name])
                for col_name in self.test_pairs.keys()
            )
        )

    def harmonize(self, other: "MatrixPairs", exclude_inconsistent_pairs: bool = True) -> "MatrixPairs":
        """Returns a new MatrixPairs object containing the common elements of two MatrixPairs objects for consistent evaluation.

        More precisely, the following operations are performed:
            - intersection of the drugs and diseases lists
            - union of the exclusion pairs
            - intersection of the test pairs
            - If exclude_inconsistent_pairs is True, exclude pairs which are in the test pairs of one but not the other
        """
        # Check if the ground truth columns are the same
        if set(self.test_pairs.keys()) != set(other.test_pairs.keys()):
            raise ValueError("The ground truth columns must the same to perform harmonization.")

        # Perform matrix harmonization
        drugs_list = self.drugs_list.join(other.drugs_list, on="source", how="inner").unique()
        diseases_list = self.diseases_list.join(other.diseases_list, on="target", how="inner").unique()
        exclusion_pairs = pl.concat([self.exclusion_pairs, other.exclusion_pairs]).unique()
        test_pairs = {
            col_name: self.test_pairs[col_name]
            .join(other.test_pairs[col_name], on=["source", "target"], how="inner")
            .unique()
            for col_name in self.test_pairs.keys()
        }

        if exclude_inconsistent_pairs:
            inconsistent_pairs = pl.concat(
                [
                    self.test_pairs[col_name].join(other.test_pairs[col_name], on=["source", "target"], how="anti")
                    for col_name in self.test_pairs.keys()
                ]
                + [
                    other.test_pairs[col_name].join(self.test_pairs[col_name], on=["source", "target"], how="anti")
                    for col_name in self.test_pairs.keys()
                ]
            ).unique()
            exclusion_pairs = pl.concat([exclusion_pairs, inconsistent_pairs]).unique()

        # Return new MatrixPairs object
        return MatrixPairs(
            drugs_list=drugs_list, diseases_list=diseases_list, exclusion_pairs=exclusion_pairs, test_pairs=test_pairs
        )


def give_matrix_pairs_from_lazyframe(
    matrix: pl.LazyFrame,
    available_ground_truth_cols: list[str],
) -> MatrixPairs:
    """Create a MatrixPairs object from a Polars LazyFrame.

    Args:
        matrix: Polars LazyFrame representing the matrix pairs.
            Columns: "source", "target", *available_ground_truth_cols
        available_ground_truth_cols: List of Boolean-valued ground truth columns.
    """
    # Extract drugs and diseases lists
    drugs_list = matrix["source"].unique()
    diseases_list = matrix["target"].unique()

    # Compute  exclusion by taking set difference with full matrix drug_list x diseases_list
    exclusion_pairs = drugs_list.join(diseases_list, how="cross").join(
        matrix.select("source", "target"), on=["source", "target"], how="inner"
    )

    # Extract test pairs
    test_pairs = {
        col_name: matrix.filter(col_name).select("source", "target") for col_name in available_ground_truth_cols
    }

    # Return a MatrixPairs object
    return MatrixPairs(
        drugs_list=drugs_list, diseases_list=diseases_list, exclusion_pairs=exclusion_pairs, test_pairs=test_pairs
    )


def harmonize_matrix_pairs(
    *matrix_pairs_all: MatrixPairs,
) -> MatrixPairs:
    """Harmonize a list of MatrixPairs objects.

    Args:
        *matrix_pairs_all: List of MatrixPairs objects to harmonize.
    """
    return reduce(lambda x, y: x.harmonize(y), matrix_pairs_all)


def all_matrix_pairs_equal(
    *matrix_pairs_all: MatrixPairs,
) -> bool:
    """Check if a list of MatrixPairs objects are equal.

    Args:
        *matrix_pairs_all: List of MatrixPairs objects to check.
    """
    return all(x == y for x, y in zip(matrix_pairs_all[0], matrix_pairs_all[1:]))
