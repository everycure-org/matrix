import pandas as pd

from typing import Dict, Any

from abc import ABC, abstractmethod
from kedro.pipeline import Pipeline, pipeline, node

from matrix.pipelines.embeddings.nodes import _bucketize


class Normalizer(ABC):
    @abstractmethod
    async def normalize(df: pd.DataFrame) -> pd.DataFrame: ...


class NCATSNormalizer(ABC):
    def __init__(self) -> None:
        pass

    async def normalize(df: pd.DataFrame) -> pd.DataFrame:
        df["normalizer"] = "abc"
        return df


def _transform(
    dfs: Dict[str, Any],
    normalizer,
):
    """Function to bucketize input data.

    Args:
        dfs: mapping of paths to df load functions
        encoder: encoder to run
    """

    def _func(dataframe: pd.DataFrame):
        return lambda df=dataframe: normalizer.normalize(df())

    shards = {}
    for path, df in dfs.items():
        # Invoke function to compute embeddings
        shards[path] = _func(df)

    return shards


def create_pipeline(**kwargs) -> Pipeline:
    """Modular pipeline to transform dataframe."""
    return pipeline(
        [
            node(
                func=_bucketize,
                inputs=["batch.raw.input"],
                outputs="batch.int.input_bucketized@spark",
                name="bucketize_input",
            ),
            node(
                func=_transform,
                inputs=["batch.int.input_bucketized@partitioned"],
                outputs="batch.int.input_transformed@partitioned",
                name="bucketize_input",
            ),
            node(
                func=lambda x, y: x.join(y, on="id", how="left"),
                inputs=["batch.int.input_transformed@spark", "batch.raw.input"],
                outputs="batch.prm.result@spark",
                name="bucketize_input",
            ),
        ]
    )
