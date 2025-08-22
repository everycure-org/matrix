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


def is_valid_biolink_category(category: str) -> bool:
    """Check if a category is valid in the biolink model using toolkit.

    Args:
        category: Category string to validate.

    Returns:
        True if category exists in biolink model, False otherwise.
    """
    return tk.is_category(category)


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

    Rules:
     1. If node has core_id (is a core entity), preserve existing category (already set correctly in transformer)
     2. If all_categories only contains "biolink:NamedThing", preserve existing category if more specific
     3. Otherwise, use the most specific category from all_categories

     Note: Expects nodes DataFrame to potentially have a 'core_id' column from core_id_mapping join

     Example:
     - node has all_categories [biolink:ChemicalEntity, biolink:NamedThing]
     - then node will be assigned biolink:ChemicalEntity as most specific category

    """

    # UDF Functions
    category_validation_udf = F.udf(is_valid_biolink_category, T.BooleanType())
    hierarchy_udf = F.udf(get_ancestors_for_category_delimited, T.ArrayType(T.StringType()))

    # For Rule 1: Ensure core_id column is present  - False if column doesn't exist, otherwise check if not null
    core_id_present = F.lit(False) if "core_id" not in nodes.columns else F.col("core_id").isNotNull()

    # For Rule 2: if all_categories only contains "biolink:NamedThing", update it using the hierarchy of the category, only if it is biolink compliant.
    namedthing_processed = nodes.filter(F.col("all_categories") == F.array(F.lit("biolink:NamedThing"))).withColumn(
        "updated_all_categories",
        F.when(
            F.col("category").isNotNull()
            & (F.col("category") != "biolink:NamedThing")
            # Check if category column has a value in it and validate against biolink model
            & category_validation_udf(F.col("category")),
            hierarchy_udf(F.col("category")),
        ).otherwise(F.col("all_categories")),
    )

    # Replace all_categories in the original DataFrame for valid categories
    nodes_with_updated_categories = (
        nodes.join(namedthing_processed.select("id", "updated_all_categories"), on="id", how="left")
        .withColumn("all_categories", F.coalesce("updated_all_categories", "all_categories"))
        .drop("updated_all_categories")
    )

    # pre-calculate the mappping table of ID -> most specific category
    more_specific_node_category_mapping = (
        nodes_with_updated_categories.select("id", "all_categories")
        .withColumn("candidate_category", F.explode("all_categories"))
        .withColumn(
            "parents",
            hierarchy_udf(F.col("candidate_category")),
        )
        # If the parents list is empty, it means the parents could not be found, hence why we are removing those rows.
        .withColumn("depth", F.array_size("parents"))
        .filter(F.col("depth") > 0)
        # Keep the row with the maximum parents, i.e. the deepest one in the hierarchy.
        .withColumn("row_num", F.row_number().over(ps.Window.partitionBy("id").orderBy(F.col("depth").desc())))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
        .select("id", F.col("candidate_category").alias("most_specific_from_all_categories"))
    )

    # Apply rules logic to determine final category
    final_nodes = nodes_with_updated_categories.join(
        more_specific_node_category_mapping, on="id", how="left"
    ).withColumn(
        "final_category",
        F.when(
            # Rule 1: Core entities (have core_id) keep their existing category
            core_id_present,
            F.col("category"),
        )
        .when(
            # Rule 2: If all_categories only contains "biolink:NamedThing", keep existing category if more specific
            (F.col("all_categories") == F.array(F.lit("biolink:NamedThing")))
            & (F.col("category") != "biolink:NamedThing")
            & (F.col("category").isNotNull()),
            F.col("category"),
        )
        .otherwise(
            # Rule 3: Use most specific from all_categories
            F.coalesce("most_specific_from_all_categories", "category")
        ),
    )

    return final_nodes.withColumn("category", F.col("final_category")).drop(
        "most_specific_from_all_categories", "final_category"
    )
