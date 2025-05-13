import logging
from typing import Any

import pyspark.sql as ps
from matrix.inject import inject_object
from pyspark.sql.functions import rand

logger = logging.getLogger(__name__)


def _apply_transformations(df: ps.DataFrame, transformations: dict[str, Any]) -> ps.DataFrame:
    """Apply a series of transformations to a DataFrame.

    This function applies a sequence of transformations to a DataFrame while maintaining
    logging information about the transformations being applied.

    Args:
        df: Input DataFrame to transform
        transformations: Dictionary of transformation instances to apply, where the key is the transformation name

    Returns:
        Transformed DataFrame
    """
    logger.info(f"Applying {len(transformations)} transformations")

    for name, transform_instance in transformations.items():
        logger.info(f"Applying transformation: {name}")
        df = transform_instance.apply(df)

    return df


@inject_object()
def apply_matrix_transformations(
    matrix: ps.DataFrame,
    transformations: dict[str, Any],
) -> ps.DataFrame:
    """Apply configured transformations to the matrix.

    Args:
        matrix: DataFrame containing matrix data
        transformations: Dictionary containing transformation configurations

    Returns:
        Transformed matrix DataFrame
    """

    return _apply_transformations(matrix, transformations)


def shuffle_and_limit_matrix(
    matrix: ps.DataFrame,
) -> ps.DataFrame:
    """Function to shuffle and limit the matrix to a subset of rows.

    Args:
        matrix: Dataframe with matrix scores
    """
    logger.info("Shuffling and limiting matrix")
    return matrix.orderBy(rand()).limit(10).orderBy("rank")
