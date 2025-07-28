import logging

import pandas as pd
import polars as pl
from matrix_validator.validator_polars import ValidatorPolarsDataFrameImpl

logger = logging.getLogger(__name__)


def validate_datasets(nodes: pd.DataFrame, edges: pd.DataFrame) -> str:
    """Function to run the Matrix Validator."""

    nodes = nodes.astype(str)
    edges = edges.astype(str)

    validator = ValidatorPolarsDataFrameImpl(nodes=pl.from_pandas(nodes), edges=pl.from_pandas(edges))
    validator.validate()
    violations = validator.violations
    if violations:
        logger.error("There were Matrix Validation violations")
        return "\n".join(violations)
    return ""
