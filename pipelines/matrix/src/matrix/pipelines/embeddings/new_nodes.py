from ray.data import Dataset
import polars as pl
from typing import Tuple


def filter_missing_embeddings(
    data: pl.LazyFrame, cache: pl.LazyFrame
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    # joins data on cache and determines which rows are missing and which
    # were already calculated and are stored in the "cache" directory
    pass


def compute_embeddings(data: Dataset, model: str) -> Dataset:
    # calls a remote function on each row of the dataset
    # assumes the column "input" contains the input data
    # column "output" will contain the result
    pass


def combine_frames(
    existing_embeddings: pl.LazyFrame, new_embeddings: pl.LazyFrame
) -> pl.LazyFrame:
    # re-joins the existing embeddings with the new embeddings
    # to produce a final dataset with a single "embeddings" column
    existing_embeddings.concat(new_embeddings)
    pass
