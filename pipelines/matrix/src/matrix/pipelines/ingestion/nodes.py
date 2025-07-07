import logging
from typing import Tuple

import pandas as pd

from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output

logger = logging.getLogger(__name__)


@check_output(
    schema=DataFrameSchema(
        columns={
            "drug|disease": Column(str, nullable=False),
            "y": Column(int, nullable=False),
        },
        unique=["source", "target", "drug|disease"],
    )
)
def create_gt(pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    """Converts the KGML-xDTD true positives and true negative dataframes into a singular dataframe compatible with EC format."""
    pos_df["indication"], pos_df["contraindication"], pos_df["y"] = True, False, 1
    neg_df["indication"], neg_df["contraindication"], neg_df["y"] = False, True, 0
    gt_df = pd.concat([pos_df, neg_df], axis=0)
    gt_df["drug|disease"] = gt_df["source"] + "|" + gt_df["target"]
    return gt_df


def create_gt_nodes_edges(edges: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    id_list = set(edges.source) | set(edges.target)
    nodes = pd.DataFrame(id_list, columns=["id"])
    edges.rename({"source": "subject", "target": "object"}, axis=1, inplace=True)
    return nodes, edges
