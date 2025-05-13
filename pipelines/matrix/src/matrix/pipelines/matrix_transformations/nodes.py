import logging

import polars as pl
import pyspark.sql as ps
from matrix.pipelines.matrix_transformations.transformations import FrequentFlyerTransformation
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

    # Apply the transformation
    return FrequentFlyerTransformation(matrix_df=matrix).apply()
