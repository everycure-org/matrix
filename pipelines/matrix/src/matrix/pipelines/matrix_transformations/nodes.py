import logging
from typing import Any

import pyspark.sql as ps
from matrix.inject import inject_object

logger = logging.getLogger(__name__)


@inject_object()
def apply_matrix_transformations(
    matrix: ps.DataFrame,
    transformations: dict[str, Any],
    score_col: str,
) -> ps.DataFrame:
    """Apply a series of transformations to a DataFrame.

    This function applies a sequence of transformations to a DataFrame while maintaining
    logging information about the transformations being applied.

    Args:
        matrix: Input DataFrame to transform
        transformations: Dictionary of transformation instances to apply, where the key is the transformation name

    Returns:
        Transformed DataFrame
    """
    logger.info(f"Applying {len(transformations)} transformations")

    df = matrix
    for name, transform_instance in transformations.items():
        logger.info(f"Applying transformation: {name}")
        df = transform_instance.apply(df, score_col)

    return df


def return_predictions(
    sorted_matrix_df: ps.DataFrame,
) -> ps.DataFrame:
    """Store the full model predictions.

    Args:
        sorted_matrix_df: DataFrame containing the sorted matrix
        **kwargs: Extra arguments such as the drug and disease lists for tables
    """

    return sorted_matrix_df
