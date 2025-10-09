import abc
from typing import List

import matplotlib.pyplot as plt
import polars as pl
from matrix.pipelines.run_comparison.input_paths import InputPathsMultiFold


class ComparisonEvaluation(abc.ABC):
    """Abstract base class for run-comparison evaluations."""

    @abc.abstractmethod
    def evaluate_single_fold(self, input_matrices: dict[str, dict[int, pl.LazyFrame]]) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate_multi_fold(self, input_matrices: dict[str, dict[int, pl.LazyFrame]]) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate_bootstrap(self, input_matrices: dict[str, dict[int, pl.LazyFrame]]) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def plot_results(results: pl.DataFrame) -> plt.Figure:
        pass


class FullMatrixRecallAtN(ComparisonEvaluation):
    """Recall@N evaluation"""

    def __init__(self, bool_test_col: str, n_max: int, perform_sort: bool = True):
        """Initialize an instance of FullMatrixRecallAtN.

        Args:
            bool_test_col: Boolean column in the matrix indicating the known positive test set.
            n_max: Maximum value of n to compute recall@n score for.
            perform_sort: Whether to sort the matrix or expect the dataframe to be sorted already.
        """
        self.bool_test_col = bool_test_col
        self.n_max = n_max
        self.perform_sort = perform_sort

    def give_recall_at_n_values(
        self,
        matrix: pl.DataFrame,
        n_lst: list[int],
        score_col_name: str,
    ) -> List[float]:
        """
        Returns the recall@n score for a list of n values.

        Args:
            matrix: Dataframe containing a single set of predictions
            n_lst: List of n values to calculate the recall@n score for.
            score_col_name: Column in the matrix containing the treat scores.

        Returns:
            A list of recall@n scores for the list of n values.
        """
        N = len(matrix.filter(pl.col(self.bool_test_col)))  # Number of known positives
        if N == 0:
            return [0] * len(n_lst)

        # Sort by score
        if self.perform_sort:
            matrix = matrix.sort(by=score_col_name, descending=True)

        # Ranks of the known positives
        ranks_series = (
            matrix.with_row_index("index").filter(pl.col(self.bool_test_col)).select(pl.col("index")).to_series() + 1
        )

        # Return Recall@n scores
        return [(ranks_series <= n).sum() / N for n in n_lst]

    def evaluate_single_fold(
        self, input_matrices: dict[str, dict[int, pl.LazyFrame]], input_paths: dict[str, InputPathsMultiFold]
    ) -> pl.DataFrame:
        """Evaluate recall@n against the provided single fold matrices.

        Args:
            input_matrices: Dictionary of polars LazyFrames of predictions and labels.
            input_path: Object containing the score column name for each model.

        Returns:
            polars DataFrame with columns `n` and `recall_at_n`.
        """
        n_lst = list(range(self.n_max))
        output_dataframe = pl.DataFrame({"n": n_lst})

        for model_name, matrices_for_model in input_matrices.items():
            # Take first fold and materialize in memory as Polars dataframe
            matrix = matrices_for_model[0].collect()

            # Compute recall@n values and join to output results dataframe
            score_col_name = input_paths[model_name].score_col_name
            recall_at_n_values = self.give_recall_at_n_values(matrix, n_lst, score_col_name)
            results_df_model = pl.DataFrame({"n": n_lst, f"recall_at_n_{model_name}": recall_at_n_values})
            output_dataframe = output_dataframe.join(results_df_model, how="left", on="n")

        return output_dataframe

    def evaluate_multi_fold(self, input_matrices: dict[str, dict[int, pl.LazyFrame]]) -> pl.DataFrame:
        return  # TODO

    def evaluate_bootstrap(
        self, input_matrices: dict[str, dict[str, pl.LazyFrame]], input_paths: InputPathsMultiFold
    ) -> pl.DataFrame:
        return  # TODO

    def plot_results(results: pl.DataFrame) -> plt.Figure:
        return  # TODO

        # recall = give_recall_at_n(matrix, n_lst, bool_test_col=self.bool_test_col, score_col=input_paths[model_name].score_col_name)

        # return pd.DataFrame({"n": n_lst, "recall_at_n": recall})
