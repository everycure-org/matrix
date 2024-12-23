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


class CountValuesQualityControl(QaulityControl):
    def run(self, df: ps.DataFrame) -> ps.DataFrame:
        return df.count()


class CountColumnValuesAggregatedQualityControl(QaulityControl):
    """Quality control suite to compute value counts."""

    # FUTURE: Split into 2 quality control classes, 1 with column and one with expr?
    def __init__(self, expr: str, filter_expr: str) -> None:
        """
        Initialize the CountColumn values quality control.

        Args:
            expr: (str) expression to apply before grouping
            filter_expr: (str) If specified, filter expression is executed _before_ counts
        """
        self._expr = F.expr(expr)
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
