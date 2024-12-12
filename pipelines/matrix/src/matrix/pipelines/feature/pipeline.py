import abc
from typing import List
import pyspark.sql.functions as F

from pyspark.sql import DataFrame
from kedro.pipeline import Pipeline, node, pipeline

from refit.v1.core.inject import inject_object


class Flag(abc.ABC):
    def __init__(self, flag: str) -> None:
        self._flag = flag

    @abc.abstractmethod
    def apply(self, df: DataFrame) -> DataFrame:
        """Function to apply flag."""
        ...


class ColumnValueIsInFlag(Flag):
    def __init__(self, flag: str, column: str, values: str) -> None:
        super().__init__(flag)
        self._column = column
        self._values = values

    def apply(self, df: DataFrame) -> DataFrame:
        return df.withColumn(self._flag, F.col(self._column).isin(self._values))


class ColumnValueExprFlag(Flag):
    def __init__(self, flag: str, expr: str) -> None:
        super().__init__(flag)
        self._expr = expr

    def apply(self, df: DataFrame) -> DataFrame:
        return df.withColumn(self._flag, F.expr(self._expr))


@inject_object()
def create_flags(df: DataFrame, instructions: List[Flag]) -> DataFrame:
    # TODO: Delete
    df = df.withColumn("publications", F.split(F.col("publications:string[]"), "\u01c2"))

    for instruction in instructions:
        df = df.transform(instruction.apply)

    return df


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    return pipeline(
        [
            node(
                inputs=["ingestion.raw.rtx_kg2.nodes@spark", "params:flag_instructions"],
                func=create_flags,
                outputs=None,
                name="create_flags",
            )
        ]
    )
