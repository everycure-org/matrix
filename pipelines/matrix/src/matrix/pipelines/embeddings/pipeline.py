"""Embeddings pipeline."""
import pandas as pd

from typing import List

from refit.v1.core.inline_has_schema import has_schema

from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline([])
