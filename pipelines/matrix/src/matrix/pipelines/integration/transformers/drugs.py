import logging
from typing import Dict

import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from .transformer import Transformer

logger = logging.getLogger(__name__)


class DrugsTransformer(Transformer):
    """Transformer for drug input source."""

    def transform(self, nodes_df: DataFrame, **kwargs) -> Dict[str, DataFrame]:
        # fmt: off
        df = (
            nodes_df
            .withColumn("id",                                f.col("curie"))
            .withColumn("name",                              f.col("curie_label"))
            .withColumn("category",                          f.lit("biolink:Drug"))
        )
        # fmt: on
        filters = [f for f in nodes_df.columns if f.startswith("is_")]
        for filter in filters:
            df = df.withColumn(filter, f.col(filter).cast("boolean"))
        return {"nodes": df}
