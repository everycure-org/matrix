import logging

import polars as pl
from matrix_validator.validator_polars import ValidatorPolarsDataFrameImpl

logger = logging.getLogger(__name__)


def validate_datasets(nodes: pl.DataFrame, edges: pl.DataFrame) -> str:
    """Function to run the Matrix Validator."""
    validator = ValidatorPolarsDataFrameImpl(nodes=nodes, edges=edges)
    validator.validate()
    violations = validator.violations
    if violations:
        logger.error("There were Matrix Validation violations")
        return "\n".join(violations)
    return ""
