"""Preprocessing pipeline."""
from kedro.pipeline import Pipeline, node, pipeline


import pandas as pd


def _print(df: pd.DataFrame) -> None:
    print(df)


def create_pipeline(**kwargs) -> Pipeline:
    """Create preprocessing pipeline."""
    return pipeline(
        [
            node(
                func=_print,
                inputs=["preprocessing.raw.nodes"],
                outputs=None,
                name="load_nodes",
            )
        ]
    )
