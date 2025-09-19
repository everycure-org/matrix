import logging

import pandas as pd
import polars as pl
import pyspark.sql as ps

from .evaluation import give_recall_at_n

logger = logging.getLogger(__name__)


def recall_at_n_table(*matrices: ps.DataFrame) -> pd.DataFrame:
    """Function to calculate recall at n and return as a structured table.


    Args:
        *matrices: Variable number of matrix DataFrames

    Returns:
        A Polars DataFrame with columns 'n' and 'recall_at_n' containing the recall@n results
    """
    # TODO: Figure out whether to change out of polars
    # Convert PySpark DataFrame to Polars for evaluation
    matrix_polars = pl.from_pandas(matrices[0].toPandas())

    # Calculate recall@n scores
    n_values = [10, 20, 50, 100, 1000, 2000, 5000, 10000]

    recall_scores = give_recall_at_n(
        matrix=matrix_polars,
        n_lst=n_values,
        bool_test_col="is_known_positive",
        score_col="treat score",
        perform_sort=False,
        out_of_matrix_mode=False,
    )

    # Create structured table
    recall_table = pl.DataFrame({"n": n_values, "recall_at_n": recall_scores})

    # Return as Pandas DataFrame
    return recall_table.to_pandas()
