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


    1. If node has core_type='drug', category must be "biolink:Drug"
    2. If node has core_type='disease', category must be "biolink:Disease"
    3. If all_categories only contains "biolink:NamedThing", preserve existing category if more specific
    4. Otherwise, use the most specific category from all_categories

    Note: Expects nodes DataFrame to have a 'core_type' column from core_id_mapping join

    Example:
    - node has all_categories [biolink:ChemicalEntity, biolink:NamedThing]
    - then node will be assigned biolink:ChemicalEntity as most specific category

    """

    # For Rule 3, we need to handle category validation and hierarchy replacement
    # First, identify nodes that only have biolink:NamedThing in all_categories
    namedthing_only_nodes = nodes.filter((F.col("all_categories") == F.array(F.lit("biolink:NamedThing"))))

    # Check if category column has a value in it and validate against biolink model
    category_validation_udf = F.udf(is_valid_biolink_category, T.BooleanType())
    hierarchy_udf = F.udf(get_ancestors_for_category_delimited, T.ArrayType(T.StringType()))

    namedthing_processed = namedthing_only_nodes.withColumn(
        "updated_all_categories",
        F.when(
            F.col("category").isNotNull()
            & (F.col("category") != "biolink:NamedThing")
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
    most_specific_mapping = (
        nodes_with_updated_categories.select("id", "all_categories")
        .withColumn("candidate_category", F.explode("all_categories"))
        .withColumn(
            "parents",
            hierarchy_udf(F.col("candidate_category")),
        )
        # Our parents list is empty if the parent could not be found, we're removing
        # these elements and ensure there is a non_null check to ensure each element
        # was found in the hierarchy
        .withColumn("depth", F.array_size("parents"))
        .filter(F.col("depth") > 0)
        .withColumn("row_num", F.row_number().over(ps.Window.partitionBy("id").orderBy(F.col("depth").desc())))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
        .select("id", F.col("candidate_category").alias("most_specific_from_all_categories"))
    )

    # Join with most specific categories
    nodes_with_all_info = nodes_with_updated_categories.join(most_specific_mapping, on="id", how="left")

    # Ensure core_type column exists (add as null if missing)
    if "core_type" not in nodes_with_all_info.columns:
        nodes_with_all_info = nodes_with_all_info.withColumn("core_type", F.lit(None).cast(T.StringType()))

    # Apply rules logic to determine final category
    final_nodes = nodes_with_all_info.withColumn(
        "final_category",
        F.when(
            # Rule 1: Core drug nodes must be biolink:Drug
            F.col("core_type").isNotNull() & (F.col("core_type") == "drug"),
            F.lit("biolink:Drug"),
        )
        .when(
            # Rule 2: Core disease nodes must be biolink:Disease
            F.col("core_type").isNotNull() & (F.col("core_type") == "disease"),
            F.lit("biolink:Disease"),
        )
        .when(
            # Rule 3: If all_categories only contains "biolink:NamedThing", keep existing category if more specific
            (F.array_size("all_categories") == 1)
            & (F.array_contains("all_categories", "biolink:NamedThing"))
            & (F.col("category") != "biolink:NamedThing")
            & (F.col("category").isNotNull()),
            F.col("category"),
        )
        .otherwise(
            # Rule 4: Use most specific from all_categories
            F.coalesce("most_specific_from_all_categories", "category")
        ),
    )

    return final_nodes.withColumn("category", F.col("final_category")).drop(
        "most_specific_from_all_categories", "core_type", "final_category"
    )
