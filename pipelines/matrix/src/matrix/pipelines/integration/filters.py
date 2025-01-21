import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.functions as f
from bmt import toolkit
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
    output = tk.get_ancestors(category, mixin=mixin)
    return output


def biolink_deduplicate_edges(r_edges_df: ps.DataFrame):
    """Function to deduplicate biolink edges.

    Knowledge graphs in biolink format may contain multiple edges between nodes. Where
    edges might represent predicates at various depths in the hierarchy. This function
    deduplicates redundant edges.

    The logic leverages the path to the predicate in the hierarchy, and removes edges
    for which "deeper" paths in the hierarchy are specified. For example: there exists
    the following edges (a)-[regulates]-(b), and (a)-[negatively-regulates]-(b). Regulates
    is on the path (regulates) whereas (regulates, negatively-regulates). In this case
    negatively-regulates is "deeper" than regulates and hence (a)-[regulates]-(b) is removed.

    Args:
        edges_df: dataframe with biolink edges
    Returns:
        Deduplicated dataframe
    """
    # Enrich edges with path to predicates in biolink hierarchy
    edges_df = r_edges_df.withColumn(
        "parents", F.udf(get_ancestors_for_category_delimited, T.ArrayType(T.StringType()))(F.col("predicate"))
    )

    # Self join to find edges that are redundant
    res = (
        edges_df.alias("A")
        .join(
            edges_df.alias("B"),
            on=[
                (f.col("A.subject") == f.col("B.subject"))
                & ((f.col("A.object") == f.col("B.object")) & (f.col("A.predicate") != f.col("B.predicate")))
            ],
            how="left",
        )
        .withColumn(
            "subpath", f.col("B.parents").isNotNull() & f.expr("forall(A.parents, x -> array_contains(B.parents, x))")
        )
        .filter(~f.col("subpath"))
        .select("A.*")
        .drop("parents")
    )
    return res


def convert_biolink_hierarchy_json_to_df(biolink_predicates, col_name: str, convert_to_pascal_case: bool):
    spark = ps.SparkSession.builder.getOrCreate()
    biolink_hierarchy = spark.createDataFrame(
        unnest_biolink_hierarchy(
            col_name,
            biolink_predicates,
            prefix="biolink:",
            convert_to_pascal_case=convert_to_pascal_case,
        )
    )

    return biolink_hierarchy


def determine_most_specific_category(nodes: ps.DataFrame, biolink_categories_df: pd.DataFrame) -> ps.DataFrame:
    """Function to retrieve most specific entry for each node.

    Example:
    - node has all_categories [biolink:ChemicalEntity, biolink:NamedThing]
    - then node will be assigned biolink:ChemicalEntity as most specific category

    """

    labels_hierarchy = convert_biolink_hierarchy_json_to_df(
        biolink_categories_df, "category", convert_to_pascal_case=True
    )

    # pre-calculate the mappping table of ID -> most specific category
    mapping_table = nodes.select("id", "all_categories").withColumn("category", F.explode("all_categories"))

    mapping_table = (
        mapping_table.join(F.broadcast(labels_hierarchy), on="category", how="left")
        # some categories are not found in the biolink hierarchy
        # we deal with failed joins by setting their parents to [] == the depth as level 0 == chosen last
        .withColumn("parents", f.coalesce("parents", f.lit(f.array())))
        .withColumn("depth", F.array_size("parents"))
        .withColumn("row_num", F.row_number().over(ps.Window.partitionBy("id").orderBy(F.col("depth").desc())))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
        .select("id", "category")
    )
    # now we can join the mapping table back to the nodes
    nodes = nodes.drop("category").join(mapping_table, on="id", how="left")

    return nodes


def remove_rows_containing_category(nodes: ps.DataFrame, categories: List[str], column: str, **kwargs) -> ps.DataFrame:
    """Function to remove rows containing a category."""
    return nodes.filter(~F.col(column).isin(categories))


def unnest_biolink_hierarchy(
    scope: str,
    predicates: List[Dict[str, Any]],
    convert_to_pascal_case: bool,
    parents: Optional[List[str]] = None,
    prefix: str = "",
):
    """Function to unnest a biolink hierarchy.

    The biolink predicates are organized in an hierarchical JSON object. To enable
    hierarchical deduplication, the JSON object is pre-processed into a flat pandas
    dataframe that adds the full path to each predicate.

    NOTE: The biolink standard is a bit confusing.
    Predicates are often written in snake_case while categories are written in PascalCase.
    This function should thus be called with the right convert_to_pascal_case flag.

    Args:
        predicates: predicates to unnest
        parents: list of parents in hierarchy
        convert_to_pascal_case: whether to convert the predicate name to pascal case
        prefix: prefix to add to the predicate name
    Returns:
        Unnested dataframe
    """

    if parents is None:
        parents = []

    slices = []
    for predicate in predicates:
        name = predicate.get("name")
        if convert_to_pascal_case:
            name = to_pascal_case(name)

        # add prefix if provided
        name = f"{prefix}{name}"

        # Recurse the children
        if children := predicate.get("children"):
            slices.append(
                unnest_biolink_hierarchy(
                    scope,
                    children,
                    parents=[*parents, name],
                    convert_to_pascal_case=convert_to_pascal_case,
                    prefix=prefix,
                )
            )

        slices.append(pd.DataFrame([[name, parents]], columns=[scope, "parents"]))

    return pd.concat(slices, ignore_index=True)


def to_pascal_case(s: str) -> str:
    # PascalCase is a writing style (like camelCase) where the first letter of each word is capitalized
    words = s.split("_")
    for i, word in enumerate(words):
        words[i] = word[0].upper() + word[1:]
    return "".join(words)
