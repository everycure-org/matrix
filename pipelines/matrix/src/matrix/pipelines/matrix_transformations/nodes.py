import logging

import polars as pl
import pyspark.sql as ps
from matrix.pipelines.matrix_transformations.transformations.frequent_flyers import give_almost_pure_transformation
from pyspark.sql.functions import rand

logger = logging.getLogger(__name__)


# TODO: Move this to params
# Call any numbr of transformations
def shuffle_and_limit_matrix(
    matrix: ps.DataFrame,
) -> ps.DataFrame:
    """Function to transform matrix to reduce the prevalence of frequent flyer drugs and diseases appreaing in top n.

    Args:
        matrix: Dataframe with untransformed matrix scores
    """

    print("Shuffle and limit matrix")

    # Drop __index_level_0__ column if it exists
    if "__index_level_0__" in matrix.columns:
        matrix = matrix.drop("__index_level_0__")

    # Return first 10 rows as a Spark DataFrame
    return matrix.orderBy(rand()).limit(10).orderBy("rank")


# TODO: Move this to params
# Call any numbr of transformations
def frequent_flyer_transformation(
    matrix: ps.DataFrame,
) -> ps.DataFrame:
    """Function to transform matrix to reduce the prevalence of frequent flyer drugs and diseases appreaing in top n.

    Args:
        matrix: Dataframe with untransformed matrix scores
    """

    print("Frequent flyer transformation")

    # Drop __index_level_0__ column if it exists
    if "__index_level_0__" in matrix.columns:
        matrix = matrix.drop("__index_level_0__")

    # Apply the transformation
    return give_almost_pure_transformation(
        matrix,
        gamma=0.05,
        epsilon=0.001,
        score_col="treat score",
        output_col="transformed_score",
        perform_sort=True,
    )
