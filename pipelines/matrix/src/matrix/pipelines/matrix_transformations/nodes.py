import logging
from typing import Any

import pyspark.sql as ps
from matrix import settings
from matrix.inject import inject_object
from pyspark.sql import functions as F

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
    known_pairs: ps.DataFrame,
) -> ps.DataFrame:
    """Store the full model predictions including training data.

    This function adds training data rows from the known_pairs DataFrame to the sorted matrix.
    The training data rows are added at the bottom of the matrix and have no scores.
    They are flagged with the is_known_positive and is_known_negative flags.

    Args:
        sorted_matrix_df: DataFrame containing the sorted matrix with predictions
        known_pairs: DataFrame containing known drug-disease pairs with fold and split information

    Returns:
        DataFrame with training data rows appended to the sorted matrix
    """

    n_cross_val_folds = settings.DYNAMIC_PIPELINES_MAPPING().get("cross_validation")["n_cross_val_folds"]

    # Filter known_pairs to get training data from the specified fold using PySpark operations
    train_data = known_pairs.filter((F.col("fold") == n_cross_val_folds) & (F.col("split") == "TRAIN")).select(
        "source", "target", "y"
    )

    # Add is_known_positive and is_known_negative flags using PySpark column operations
    train_data = train_data.withColumn("is_known_positive", F.col("y") == 1)
    train_data = train_data.withColumn("is_known_negative", F.col("y") == 0)

    result_df = sorted_matrix_df.unionByName(train_data, allowMissingColumns=True)

    return result_df
