from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key
from refit.v1.core.output_primary_key import _duplicate_and_null_check


@has_schema(
    schema={
        "id": "string",
        "model": "string",
        "scope": "string",
        "embedding": "array<double>",
    },
    output=0,
)
# NOTE: This validates the new cache shard does _not_
# contain duplicates. This only checks a _single_ shard though,
# hence why we're also validating the result dataframe below.
# Let's avoid using the primary key statement on the cache, as that
# might be a very heavy operation.
@primary_key(primary_key=["model", "scope", "id"], output=0)
# @inline_primary_key(primary_key=["model", "scope", "id"], df="cache")
def cache(
    dataframe: DataFrame,
    cache: DataFrame,
    model: str = "gpt-4",
    scope: str = "rtx_kg2",  # do we strictly need the scope?
    id_column: str = "id:ID",
) -> Tuple[DataFrame, DataFrame]:
    """Function to enrich embeddings with caching mechanism.
    The caching mechanism maintains a SparkDataset in append mode, thereby
    allowing it to evolve over time. It partitions the dataset according
    to the given scope and model, thereby speeding up the join process.
    WARNING: Still requires further dealing with race conditions, i.e.,
    when two pipelines with the same scope/model are running simultaneously
    this may create duplicate elements in the cache.
    FUTURE: Generalize technique to be able to deal with any call
    of GenAI interaction.
    Args:
        dataframe: dataframe to enrich
        cache: caching dataframe
        model: model being applied
        scope: embedding enrichment scope
        id_column: element providing stable identifiers for input elements
    """

    # Grab relevant cache subset
    cache_df = (
        cache.alias("cache")
        .filter(F.col("scope") == F.lit(scope))
        .filter(F.col("model") == F.lit(model))
        .join(dataframe.alias("df"), on=[F.col(id_column) == F.col("cache.id")], how="right")
    )

    # Compute embeddings for cache misses
    cache_misses = (
        cache_df.filter(F.col("embedding").isNull())
        .withColumn("scope", F.lit(scope))
        .withColumn("model", F.lit(model))
        # TODO: Enrich actual embedding here, e.g., using Ray
        .withColumn("embedding", F.lit([float(0.1), float(0.02)]))
    )

    # Compute result by combining hits and misses
    result = (
        cache_df.filter(F.col("embedding").isNotNull())
        .unionByName(cache_misses, allowMissingColumns=False)
        .select("df.*", "embedding")
    )

    # Perform duplicate on result dataframe, this would signal join explosions.
    _duplicate_and_null_check(result, [id_column], nullable=False, df_name=None)
    print(f"cache hits {dataframe.count() - cache_misses.count()} out of {dataframe.count()}")

    return [cache_misses.select("model", "scope", "embedding", id_column).withColumnRenamed(id_column, "id"), result]
