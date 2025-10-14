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
            .withColumn("category",                          f.lit("biolink:Drug"))
            .withColumnRenamed("id", "ec_id")
            .withColumnRenamed("translator_id", "id")
        )
        # fmt: on
        return {"nodes": df}
