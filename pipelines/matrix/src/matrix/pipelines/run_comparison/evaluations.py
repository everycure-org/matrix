import abc
from typing import Iterable, List, Optional

import pandas as pd
import polars as pl
import pyspark.sql as ps


class ComparisonEvaluation(abc.ABC):
    """Abstract base class for run-comparison evaluations."""

    @abc.abstractmethod
    def evaluate(self, matrices: List[ps.DataFrame]) -> pd.DataFrame:
        pass


class RecallAtN(ComparisonEvaluation):
    """Recall@N evaluation"""

    def __init__(self, n_values: Optional[Iterable[int]] = None):
        self.n_values: List[int] = list(n_values) if n_values is not None else [10, 20, 50, 100]

    def evaluate(self, matrices: List[ps.DataFrame]) -> pd.DataFrame:
        """Evaluate recall@n against the provided matrix.

        Args:
            matrix: PySpark DataFrame of predictions and labels.

        Returns:
            pandas DataFrame with columns `n` and `recall_at_n`.
        """
        # Just doing one matrix
        matrix_pl = pl.from_pandas(matrices[0].toPandas())

        n_lst: List[int] = self.n_values
        bool_test_col: str = "is_known_positive"
        score_col: str = "treat score"
        perform_sort: bool = True
        out_of_matrix_mode: bool = False

        # Number of known positives
        N = len(matrix_pl.filter(pl.col(bool_test_col)))
        if N == 0:
            return [0] * len(n_lst)

        if out_of_matrix_mode:
            matrix_pl = matrix_pl.filter(pl.col("in_matrix") | pl.col(bool_test_col))

        # Sort by treat score
        if perform_sort or out_of_matrix_mode:
            matrix_pl = matrix_pl.sort(by=score_col, descending=True)

        # Ranks of the known positives
        ranks_series = (
            matrix_pl.with_row_index("index").filter(pl.col(bool_test_col)).select(pl.col("index")).to_series() + 1
        )

        # Recall@n scores
        recall_lst = [(ranks_series <= n).sum() / N for n in n_lst]

        return pd.DataFrame({"n": n_lst, "recall_at_n": recall_lst})
