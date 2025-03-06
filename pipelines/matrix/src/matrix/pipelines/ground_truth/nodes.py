import logging
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T
from joblib import Memory
from pyspark.sql.window import Window

from matrix.inject import inject_object
from matrix.pipelines.integration.nodes import _union_datasets

# from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output

# from .schema import BIOLINK_KG_EDGE_SCHEMA, BIOLINK_KG_NODE_SCHEMA

# TODO move these into config
memory = Memory(location=".cache/nodenorm", verbose=0)
logger = logging.getLogger(__name__)


# @check_output(
#     DataFrameSchema(
#         columns={
#             "id": Column(T.StringType(), nullable=False),
#         },
#         unique=["id"],
#     ),
#     df_name="nodes",
# )
# @check_output(
#     DataFrameSchema(
#         columns={
#             "subject": Column(T.StringType(), nullable=False),
#             "predicate": Column(T.StringType(), nullable=False),
#             "object": Column(T.StringType(), nullable=False),
#         },
#         # removing the uniqueness constraint as some KGs have duplicate edges. These will be deduplicated later when we do edge deduplication anyways
#         # unique=["subject", "predicate", "object"],
#     ),
#     df_name="edges",
#     raise_df_undefined=False,
# )
@inject_object()
def transform(transformer, **kwargs) -> Dict[str, ps.DataFrame]:
    return transformer.transform(**kwargs)


# @check_output(
#     schema=BIOLINK_KG_EDGE_SCHEMA,
#     pass_columns=True,
# )
def unify_edges(
    *edges,
) -> ps.DataFrame:
    """Function to unify edges datasets."""
    # fmt: off
    return _union_datasets(*edges)
