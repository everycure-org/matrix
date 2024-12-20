import abc


import pyspark.sql as ps

from pyspark.sql import functions as F
from pyspark.sql import types as T


class QaulityControl(abc.ABC):
    """Base class for computing quality control metrics."""

    @abc.abstractmethod
    def run(df: ps.DataFrame) -> ps.DataFrame:
        """Function to run evaluation suite on given dataset.

        Args:
            df: Dataframe.
        Returns:
            Dataframe with quality control metrics.
        """
        ...


class CountColumnValuesQualityControl(QaulityControl):
    """Quality control suite to compute value counts."""

    # FUTURE: Split into 2 quality control classes, 1 with column and one with expr?
    def __init__(self, column: str, expr: str, filter_expr: str) -> None:
        """
        Initialize the CountColumn values quality control.

        Args:
            column: (str) Column to perform counts on
            expr: (str) If specified, output of expression will be used to count on
            filter_expr: (str) If specified, filter expression is executed _before_ counts
        """
        if expr:
            self._expr = F.expr(expr)
        else:
            # If no expression specified, we fallaback to the input column
            self._expr = F.col(column)

        self._filter_expr = filter_expr

    def run(self, df: ps.DataFrame) -> ps.DataFrame:
        return (
            df.withColumn("_col", self._expr)
            .transform(self._cond_filter, self._filter_expr)
            .groupBy("_col")
            .count()
            .select(
                F.col("_col").cast(T.StringType()).alias("metric"), F.col("count").cast(T.IntegerType()).alias("value")
            )
        )

    @staticmethod
    def _cond_filter(df: ps.DataFrame, filter_expr):
        if filter_expr:
            return df.filter(filter_expr)
        else:
            return df


# TODO: Define other logical suites here
