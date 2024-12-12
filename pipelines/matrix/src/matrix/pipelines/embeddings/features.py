import abc
from typing import List
import pyspark.sql.functions as F

from pyspark.sql import DataFrame

from refit.v1.core.inject import inject_object


class Transform(abc.ABC):
    """Base class to represent generic transform."""

    @abc.abstractmethod
    def apply(self, df: DataFrame) -> DataFrame:
        """Function to apply flag."""
        ...


class Filter(Transform, abc.ABC):
    """Base class to represent filtering transformations."""

    ...


class HasNotFlagsFilter(Filter):
    """Class to represent filtering based on flags."""

    def __init__(self, flags: List[str]) -> None:
        self._flags = flags

    def apply(self, df: DataFrame) -> DataFrame:
        return df.filter(" AND ".join([f"{flag} IS FALSE" for flag in self._flags]))


class Flag(Transform, abc.ABC):
    """Base class to represent flags."""

    def __init__(self, flag: str) -> None:
        self._flag = flag


class ColumnValueIsInFlag(Flag):
    """Base class to represent flags."""

    def __init__(self, flag: str, column: str, values: str) -> None:
        super().__init__(flag)
        self._column = column
        self._values = values

    def apply(self, df: DataFrame) -> DataFrame:
        return df.withColumn(self._flag, F.col(self._column).isin(self._values))


class ColumnValuesOverlapWithFlag(Flag):
    def __init__(self, flag: str, column: str, values: str) -> None:
        super().__init__(flag)
        self._column = column
        self._values = values

    def apply(self, df: DataFrame) -> DataFrame:
        return df.withColumn(self._flag, F.arrays_overlap(F.col(self._column), F.lit(self._values)))


class HasFlags(Flag):
    def __init__(self, flag: str, flags: List[str]) -> None:
        super().__init__(flag)
        self._flags = flags

    def apply(self, df: DataFrame) -> DataFrame:
        return df.withColumn(self._flag, F.expr(" AND ".join([f"{flag} IS TRUE" for flag in self._flags])))


class HasAnyFlag(Flag):
    def __init__(self, flag: str, flags: List[str]) -> None:
        super().__init__(flag)
        self._flags = flags

    def apply(self, df: DataFrame) -> DataFrame:
        return df.withColumn(self._flag, F.expr(" OR ".join([f"{flag} IS TRUE" for flag in self._flags])))


@inject_object()
def apply_transforms(df: DataFrame, transformations: List[Transform]) -> DataFrame:
    for transformation in transformations:
        df = df.transform(transformation.apply)

    return df
