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
            .withColumn("category", f.lit("biolink:Disease"))
            .drop("synonyms") # Dropped because a) unused and b) causes issues later. Details in issue ECDATA-831
        )
        # fmt: on
        return {"nodes": df}
