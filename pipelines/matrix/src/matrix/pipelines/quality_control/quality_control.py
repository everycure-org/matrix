import abc

import pyspark.sql as ps


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

    def __init__(self, column: str) -> None:
        self._column = column

    def run(self, df: ps.DataFrame) -> ps.DataFrame:
        return df.groupBy(self._column).count()


# TODO: Define other logical suites here
