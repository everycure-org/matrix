import logging
from typing import List

import pyspark.sql as ps
import pyspark.sql.functions as F
from bmt import toolkit
from matrix_schema.utils.pandera_utils import Column, DataFrameSchema, check_output
from pyspark.sql import types as T

tk = toolkit.Toolkit()

logger = logging.getLogger(__name__)


def get_ancestors_for_category_delimited(category: str, mixin: bool = False) -> List[str]:
    """Wrapper function to get ancestors for a category. The arguments were used to match the args used by Chunyu
    https://biolink.github.io/biolink-model-toolkit/index.html#bmt.toolkit.Toolkit.get_ancestors

    Args:
        category: Category to get ancestors for
        formatted: Whether to format element names as curies
        mixin: Whether to include mixins
        reflexive: Whether to include query element in the list
    Returns:
        List of ancestors in a string format
    """
    return tk.get_ancestors(category, mixin=mixin, formatted=True, reflexive=True)


@check_output(
    DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "category": Column(T.StringType(), nullable=False),
        },
        unique=["id"],
    ),
)
def determine_most_specific_category(nodes: ps.DataFrame) -> ps.DataFrame:
    """Function to retrieve most specific entry for each node.

    This function uses the `all_categories` column to infer a final `category` for the
    node based on the category that is the deepest in the hierarchy. We remove any categories
    from `all_categories` that could not be resolved against biolink.

    Example:
    - node has all_categories [biolink:ChemicalEntity, biolink:NamedThing]
    - then node will be assigned biolink:ChemicalEntity as most specific category

    """
    # pre-calculate the mappping table of ID -> most specific category
    mapping_table = (
        nodes.select("id", "all_categories")
        .withColumn("category", F.explode("all_categories"))
        .withColumn(
            "parents", F.udf(get_ancestors_for_category_delimited, T.ArrayType(T.StringType()))(F.col("category"))
        )
        # Our parents list is empty if the parent could not be found, we're removing
        # these elements and ensure there is a non_null check to ensure each element
        # was found in the hierarchy
        .withColumn("depth", F.array_size("parents"))
        .filter(F.col("depth") > 0)
        .withColumn("row_num", F.row_number().over(ps.Window.partitionBy("id").orderBy(F.col("depth").desc())))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
        .select("id", "category")
    )

    return nodes.drop("category").join(mapping_table, on="id", how="left")
