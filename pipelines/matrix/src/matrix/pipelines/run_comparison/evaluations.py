import abc
from typing import List

import polars as pl
from matrix.pipelines.run_comparison.input_paths import InputPathsMultiFold


class ComparisonEvaluation(abc.ABC):
    """Abstract base class for run-comparison evaluations."""

    @abc.abstractmethod
    def evaluate(self, input_matrices: dict[str, dict[str, pl.LazyFrame]]) -> pl.DataFrame:
        pass


class FullMatrixRecallAtN(ComparisonEvaluation):
    """Recall@N evaluation"""

    def __init__(self, bool_test_col: str, n_max: int):
        self.bool_test_col = bool_test_col
        self.n_max = n_max

    def evaluate(
        self, input_matrices: dict[str, dict[str, pl.LazyFrame]], input_paths: InputPathsMultiFold
    ) -> pl.DataFrame:
        """Evaluate recall@n against the provided matrix.

        Args:
            matrices: list of polars LazyFrames of predictions and labels.
            input_path: Object containing the score column name for each model.

        Returns:
            polars DataFrame with columns `n` and `recall_at_n`.
        """

        n_lst = list(range(n_max))
        return None

        # recall = give_recall_at_n(matrix, n_lst, bool_test_col=self.bool_test_col, score_col=input_paths[model_name].score_col_name)

        # return pd.DataFrame({"n": n_lst, "recall_at_n": recall})


# Direct copy-paste from lab-notebooks
def give_recall_at_n(
    matrix: pl.DataFrame,
    n_lst: list[int],
    bool_test_col: str,
    score_col: str,
    perform_sort: bool = True,
    out_of_matrix_mode: bool = False,
) -> List[float]:
    """
    Returns the recall@n score for a list of n values.

    Args:
        matrix: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrices.
        n_lst: List of n values to calculate the recall@n score for.
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.
        perform_sort: Whether to sort the matrix by the treat score, or expect the dataframe to be sorted already.
        out_of_matrix_mode: Whether to use the out of matrix mode, where pairs outside the matrix may be used in the calculation.
            In this case, the matrix dataframe must also contain a boolean column "in_matrix".
    Returns:
        A list of recall@n scores for the list of n values.
    """
    # We can figure out where to convert to polars in the future
    matrix = pl.from_pandas(matrix.toPandas())
    # Number of known positives
    N = len(matrix.filter(pl.col(bool_test_col)))
    if N == 0:
        return [0] * len(n_lst)

    if out_of_matrix_mode:
        matrix = matrix.filter(pl.col("in_matrix") | pl.col(bool_test_col))

    # Sort by treat score
    if perform_sort or out_of_matrix_mode:
        matrix = matrix.sort(by=score_col, descending=True)

    # Ranks of the known positives
    ranks_series = matrix.with_row_index("index").filter(pl.col(bool_test_col)).select(pl.col("index")).to_series() + 1

    # Recall@n scores
    recall_lst = [(ranks_series <= n).sum() / N for n in n_lst]

    return recall_lst
