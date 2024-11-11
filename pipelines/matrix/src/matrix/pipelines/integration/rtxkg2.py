import logging
from typing import Dict

import pandera.pyspark as pa
import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import DataFrame
from refit.v1.core.inline_primary_key import primary_key

from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema, cols_for_schema

logger = logging.getLogger(__name__)


RTX_SEPARATOR = "\u01c2"


@pa.check_output(KGNodeSchema)
def transform_rtxkg2_nodes(nodes_df: DataFrame) -> DataFrame:
    """Transform RTX KG2 nodes to our target schema.

    Args:
        nodes_df: Nodes DataFrame.

    Returns:
        Transformed DataFrame.
    """
    # fmt: off
    return (
        nodes_df
        .withColumn("upstream_data_source",              f.array(f.lit("rtxkg2")))
        .withColumn("labels",                            f.split(f.col(":LABEL"), RTX_SEPARATOR))
        .withColumn("all_categories",                    f.split(f.col("all_categories:string[]"), RTX_SEPARATOR))
        .withColumn("all_categories",                    f.array_distinct(f.concat("labels", "all_categories")))
        .withColumn("equivalent_identifiers",            f.split(f.col("equivalent_curies:string[]"), RTX_SEPARATOR))
        .withColumn("publications",                      f.split(f.col("publications:string[]"), RTX_SEPARATOR))
        .withColumn("international_resource_identifier", f.col("iri"))
        .withColumnRenamed("id:ID", "id")
        .select(*cols_for_schema(KGNodeSchema))
    )
    # fmt: on


@pa.check_output(KGEdgeSchema)
def transform_rtxkg2_edges(edges_df: DataFrame, curie_to_pmids: DataFrame, semmed_filters: Dict[str, str]) -> DataFrame:
    """Transform RTX KG2 edges to our target schema.

    Args:
        edges_df: Edges DataFrame.
        pubmed_mapping: pubmed mapping
    Returns:
        Transformed DataFrame.
    """

    # fmt: off
    return (
        edges_df
        .withColumn("upstream_data_source",          f.array(f.lit("rtxkg2")))
        .withColumn("knowledge_level",               f.lit(None).cast(T.StringType()))
        .withColumn("aggregator_knowledge_source",   f.split(f.col("knowledge_source:string[]"), RTX_SEPARATOR)) # RTX KG2 2.10 does not exist
        .withColumn("primary_knowledge_source",      f.col("aggregator_knowledge_source").getItem(0)) # RTX KG2 2.10 `primary_knowledge_source``
        .withColumn("publications",                  f.split(f.col("publications:string[]"), RTX_SEPARATOR))
        .withColumn("subject_aspect_qualifier",      f.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
        .withColumn("subject_direction_qualifier",   f.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
        .withColumn("object_aspect_qualifier",       f.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
        .withColumn("object_direction_qualifier",    f.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
        .select(*cols_for_schema(KGEdgeSchema))
    ).transform(filter_semmed, curie_to_pmids, **semmed_filters)
    # fmt: on


@primary_key(df="curie_to_pmids", primary_key=["curie"])
def filter_semmed(
    edges_df: DataFrame,
    curie_to_pmids: DataFrame,
    publication_threshold: int,
    ngd_threshold: float,
    limit_pmids: int,
) -> DataFrame:
    """Function to filter semmed edges.

    Function that performs additional cleaning on RTX edges obtained by SemMedDB. This
    replicates preprocessing that Chuynu has been doing. Needs future refinement.

    Args:
        edges_df: Dataframe with edges
        curie_to_pmids: Dataframe mapping curies to PubMed Identifiers (PMIDs)
        publication_threshold: Threshold for publications
        ngd_threshold: threshold for ngd
    Returns
        Filtered dataframe
    """
    logger.info("Filtering semmed edges")
    logger.info(f"Number of edges: {edges_df.count()}")
    curie_to_pmids = (
        curie_to_pmids.withColumn("pmids", f.from_json("pmids", T.ArrayType(T.IntegerType())))
        # .withColumn("pmids", f.sort_array(f.col("pmids")))
        # .withColumn("limited_pmids", f.slice(f.col("pmids"), 1, limit_pmids))
        # .drop("pmids")
        # .withColumnRenamed("limited_pmids", "pmids")
        .withColumn("num_pmids", f.array_size(f.col("pmids")))
        .withColumnRenamed("curie", "id")
        .persist()
    )

    table = f.broadcast(curie_to_pmids)

    semmed_edges = (
        edges_df.alias("edges")
        .filter(f.col("primary_knowledge_source") == f.lit("infores:semmeddb"))
        # Enrich subject pubmed identifiers
        .join(
            table.alias("subj"),
            on=[f.col("edges.subject") == f.col("subj.id")],
            how="left",
        )
        # Enrich object pubmed identifiers
        .join(
            table.alias("obj"),
            on=[f.col("edges.object") == f.col("obj.id")],
            how="left",
        )
        .transform(compute_ngd)
        .withColumn("num_publications", f.size(f.col("publications")))
        # fmt: off
        .filter(
            # Retain only semmed edges more than 10 publications or ndg score below 0.6
            (f.col("num_publications") >= f.lit(publication_threshold)) & (f.col("ngd") <= f.lit(ngd_threshold))
        )
        # fmt: on
        .select("edges.*")
    )

    edges_filtered = edges_df.filter(f.col("primary_knowledge_source") != f.lit("infores:semmeddb")).unionByName(
        semmed_edges
    )
    logger.info(f"Number of edges after filtering: {edges_filtered.count()}")
    logger.info(f"Number of semmed edges after filtering: {semmed_edges.count()}")
    return edges_filtered


def compute_ngd(df: DataFrame, num_pairs: int = 3.7e7 * 20) -> DataFrame:
    """
    PySpark transformation to compute Normalized Google Distance (NGD).

    Args:
        df: Dataframe
        num_pairs: num_pairs
    Returns:
        Dataframe with ndg score
    """
    # we perform a array intersection calculation leveraging join and group by instead of array_intersect as some of the arrays are large (20M+)
    subject_side = df.select("edges.*", "subj.pmids").withColumn("pmid", f.explode("subj.pmids")).drop("subj.pmids")
    object_side = df.select("edges.*", "obj.pmids").withColumn("pmid", f.explode("obj.pmids")).drop("obj.pmids")
    joined = subject_side.join(
        object_side, on=["pmid", "edges.subject", "edges.predicate", "edges.object"], how="inner"
    )
    joined = (
        joined.groupBy("edges.subject", "edges.predicate", "edges.object")
        .agg(f.count("pmid").alias("num_common_pmids"))
        .select("edges.subject", "edges.predicate", "edges.object", "num_common_pmids")
    )
    df = df.join(joined, on=["edges.subject", "edges.predicate", "edges.object"], how="left")
    df = (
        # Take first max_pmids elements from each array
        df.withColumn(
            "ngd",
            (
                f.greatest(f.log2(f.col("subj.num_pmids")), f.log2(f.col("obj.num_pmids")))
                - f.log2(f.col("num_common_pmids"))
            )
            / (f.log2(f.lit(num_pairs)) - f.least(f.log2(f.col("subj.num_pmids")), f.log2(f.col("obj.num_pmids")))),
        )
    )
    return df
