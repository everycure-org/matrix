import logging
from typing import Dict

import pyspark.sql as ps
import pyspark.sql.functions as f

from .transformer import Transformer

logger = logging.getLogger(__name__)


class DiseasesTransformer(Transformer):
    """Transformer for disease input source."""

    def transform(self, nodes_df: ps.DataFrame, **kwargs) -> Dict[str, ps.DataFrame]:
        # fmt: off
        df = (
            nodes_df
            .withColumn("id",       f.col("category_class"))
            .withColumn("name",     f.col("label"))
            .withColumn("category", f.lit("biolink:Disease"))
        )
        filters = [f for f in nodes_df.columns if f.startswith("is_")]
        filters.append('official_matrix_filter') 
        for filter in filters:
            df = df.withColumn(filter, f.col(filter).cast("boolean"))
        # fmt: on
        return {"nodes": df}
