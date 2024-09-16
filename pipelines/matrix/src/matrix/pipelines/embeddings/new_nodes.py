from ray.data import Dataset
import polars as pl
from typing import Tuple
from pyspark.sql import DataFrame


def filter_missing_embeddings(
    data: DataFrame, cache: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    # TODO hack
    tuple(data.randomSplit([0.5, 0.5]))
    # joins data on cache and determines which rows are missing and which
    # were already calculated and are stored in the "cache" directory
    pass


def compute_embeddings(data: Dataset, model: str) -> Dataset:
    # calls a remote function on each row of the dataset
    # assumes the column "input" contains the input data
    # column "output" will contain the result
    pass


def combine_frames(
    existing_embeddings: DataFrame, new_embeddings: DataFrame
) -> DataFrame:
    # re-joins the existing embeddings with the new embeddings
    # to produce a final dataset with a single "embeddings" column
    existing_embeddings.union(new_embeddings)
    pass


@ray.remote
def _compute_embedding(text: str):
    # TODO: use a real embedding model
    return np.random.rand(512)


# def remote_to_iterator(obj_futures):
#     while obj_futures:
#         done, obj_futures = ray.wait(obj_futures)
#         yield ray.get(done[0])


def process_batch(df: pl.DataFrame) -> pl.DataFrame:
    print(f"batch size: {df.height}")
    print(df.columns)
    # TODO: using a dummy model for now
    model = get_sanitized_model(MODEL)
    # embedding_futures = [compute_embedding.remote(text) for text in tqdm(df["data"])]
    embedding_futures = [
        _compute_embedding.remote(text) for text in tqdm(df["data"], desc="submitting")
    ]
    print("waiting for embeddings calculations to complete...")
    embeddings = list(tqdm(remote_to_iterator(embedding_futures), desc="waiting..."))
    new_col = pl.Series(embeddings)
    return df.with_columns(result=new_col, model=pl.lit(model))
