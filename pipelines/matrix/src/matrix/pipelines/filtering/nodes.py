from enum import Enum

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
from rich.console import Console

from matrix.utils.pandera_utils import check_output

from ..integration.schema import BIOLINK_KG_NODE_SCHEMA


class KGSource(str, Enum):
    ROBOKOP = "robokop"
    RTXKG2 = "rtxkg2"


console = Console()


# TODO: Add check_output once defined
# @check_output(
#     schema=BIOLINK_KG_NODE_SCHEMA,
#     pass_columns=True,
# )
def _filter_kg_on_source(df: ps.DataFrame, source: KGSource) -> pd.DataFrame:
    """Function to filter KG to only the desired data source(s).

    Args:
        df: Spark DataFrame containing knowledge graph data
        source: Knowledge graph source, either 'robokop' or 'rtxkg2'

    Returns:
        Filtered pandas DataFrame
    """
    filtered = df.filter(F.array_contains("upstream_data_source", source)).limit(10)
    # Temp - convert to pandas to store csv locally
    return filtered.toPandas()


def filter_kg(
    nodes: ps.DataFrame, edges: ps.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function to filter KG to only the desired data source(s)."""

    rtxkg2_nodes = _filter_kg_on_source(nodes, "rtxkg2")
    rtxkg2_edges = _filter_kg_on_source(edges, "rtxkg2")
    robokop_nodes = _filter_kg_on_source(nodes, "robokop")
    robokop_edges = _filter_kg_on_source(edges, "robokop")

    return rtxkg2_nodes, rtxkg2_edges, robokop_nodes, robokop_edges
