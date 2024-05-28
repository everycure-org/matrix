"""Embeddings pipeline."""
import pandas as pd

from typing import List

from refit.v1.core.inline_has_schema import has_schema

from kedro.pipeline import Pipeline, node, pipeline


@has_schema(
    schema={"id": "object", "embedding": "object"}, allow_subset=True, relax=True
)
def create_int_embeddings(raw_embeddings: List) -> pd.DataFrame:
    """Function to create int embeddings."""
    return pd.DataFrame(raw_embeddings.items(), columns=["id", "embedding"])


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            node(
                func=create_int_embeddings,
                inputs=["raw.embeddings"],
                outputs="int.embeddings",
                name="create_int_embeddings",
            ),
        ]
    )
