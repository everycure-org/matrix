import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.functions as f
from bmt import toolkit
from pyspark.sql import types as T

from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output

tk = toolkit.Toolkit()

logger = logging.getLogger(__name__)


def remove_rows_containing_category(
    nodes: ps.DataFrame, categories: List[str], column: str, exclude_sources: Optional[List[str]] = None, **kwargs
) -> ps.DataFrame:
    """Function to remove rows containing a category."""
    if exclude_sources is None:
        exclude_sources = []

    df = nodes.withColumn("_exclude", f.arrays_overlap(f.col("upstream_data_source"), f.lit(exclude_sources))).filter(
        (F.col("_exclude") | ~F.col(column).isin(categories))
    )
    return df.drop("_exclude")
