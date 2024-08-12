"""mlflow decorator PoC."""
import mlflow
import functools
from pandas import DataFrame as pandasDataFrame
from pyspark.sql import DataFrame as sparkDataFrame


def mlflow_log(log_name: str, context_name: str):
    """Decorator to log inputs for mlflow ."""

    def decorator(func):
        """Decorator."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper."""
            result = func(*args, **kwargs)
            if isinstance(result, pandasDataFrame):
                mlflow.log_input(
                    mlflow.data.from_pandas(result, name=log_name), context=context_name
                )
            elif isinstance(result, sparkDataFrame):
                # need to convert it to pandas as wehn we log spark DataFrame, the digest id keeps changing
                # I suspect mlflow might be calling sparksession each time its logging spark dataframe which might have
                # different seed -> different digest id
                mlflow.log_input(
                    mlflow.data.from_pandas(result.toPandas(), name=log_name),
                    context=context_name,
                )

            return result

        return wrapper

    return decorator
