import logging
from typing import Any

import pyspark.sql as ps
from matrix.inject import inject_object

logger = logging.getLogger(__name__)


@inject_object()
def apply_matrix_transformations(
    matrix: ps.DataFrame,
    transformations: dict[str, Any],
) -> ps.DataFrame:
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
        df = transform_instance.apply(matrix)

    return df
