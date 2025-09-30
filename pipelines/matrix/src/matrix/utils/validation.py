import logging

import polars as pl
from matrix_validator.validator_polars import ValidatorPolarsDataFrameImpl

logger = logging.getLogger(__name__)


def validate(nodes: pl.DataFrame, edges: pl.DataFrame, strict: bool = False) -> str:
    """Function to run the Matrix Validator."""
    validator = ValidatorPolarsDataFrameImpl(nodes=nodes, edges=edges)
    validator.validate()
    violations = validator.violations
    if violations:
        logger.error("There were Matrix Validation violations")
        if strict:
            raise Exception("There were Matrix Validation violations. Please remedy before continuing.")
        return "\n".join(violations)
    return ""
