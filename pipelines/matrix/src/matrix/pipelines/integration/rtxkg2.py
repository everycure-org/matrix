import logging
from typing import Dict

import pandera
from pandera.pyspark import DataFrameModel, Field

import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

from .transformer import GraphTransformer

from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema, cols_for_schema

logger = logging.getLogger(__name__)


RTX_SEPARATOR = "\u01c2"


class RTXTransformer(GraphTransformer):
    @pandera.check_output(KGNodeSchema)
    def transform_nodes(self, nodes_df: pyspark.sql.DataFrame, **kwargs) -> pyspark.sql.DataFrame:
        """Transform RTX KG2 nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        return (
            nodes_df
            .withColumn("upstream_data_source",              F.array(F.lit("rtxkg2")))
            .withColumn("labels",                            F.split(F.col(":LABEL"), RTX_SEPARATOR))
            .withColumn("all_categories",                    F.split(F.col("all_categories:string[]"), RTX_SEPARATOR))
            .withColumn("equivalent_identifiers",            F.split(F.col("equivalent_curies:string[]"), RTX_SEPARATOR))
            .withColumn("publications",                      F.split(F.col("publications:string[]"), RTX_SEPARATOR).cast(T.ArrayType(T.StringType())))
            .withColumn("international_resource_identifier", F.col("iri"))
            .withColumnRenamed("id:ID", "id")
            .select(*cols_for_schema(KGNodeSchema))
        )
        # fmt: on

    @pandera.check_output(KGEdgeSchema)
    def transform_edges(
        self,
        edges_df: pyspark.sql.DataFrame,
        curie_to_pmids: pyspark.sql.DataFrame,
        semmed_filters: Dict[str, str],
        **kwargs,
    ) -> pyspark.sql.DataFrame:
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
            .withColumn("upstream_data_source",          F.array(F.lit("rtxkg2")))
            .withColumn("knowledge_level",               F.lit(None).cast(T.StringType()))
            .withColumn("aggregator_knowledge_source",   F.split(F.col("knowledge_source:string[]"), RTX_SEPARATOR)) # RTX KG2 2.10 does not exist
            .withColumn("primary_knowledge_source",      F.col("aggregator_knowledge_source").getItem(0)) # RTX KG2 2.10 `primary_knowledge_source``
            .withColumn("publications",                  F.split(F.col("publications:string[]"), RTX_SEPARATOR))
            .withColumn("subject_aspect_qualifier",      F.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
            .withColumn("subject_direction_qualifier",   F.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
            .withColumn("object_aspect_qualifier",       F.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
            .withColumn("object_direction_qualifier",    F.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
            .select(*cols_for_schema(KGEdgeSchema))
        ).transform(filter_semmed, curie_to_pmids, **semmed_filters)
        # fmt: on


class CurieToPMIDsSchema(DataFrameModel):
    """Schema for a curie to pmids mapping."""

    # fmt: off
    curie:          T.StringType()                  = Field(nullable=False)  # type: ignore
    # fmt: on

    class Config:
        strict = False
        unique = ["curie"]


@pandera.check_input(CurieToPMIDsSchema, obj_getter="curie_to_pmids")
def filter_semmed(
    edges_df: pyspark.sql.DataFrame,
    curie_to_pmids: pyspark.sql.DataFrame,
    publication_threshold: int,
    ngd_threshold: float,
    limit_pmids: int,
) -> pyspark.sql.DataFrame:
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
    curie_to_pmids = (
        curie_to_pmids.withColumn("pmids", F.from_json("pmids", T.ArrayType(T.IntegerType())))
        .withColumn("pmids", F.sort_array(F.col("pmids")))
        .withColumn("limited_pmids", F.slice(F.col("pmids"), 1, limit_pmids))
        .drop("pmids")
        .withColumnRenamed("limited_pmids", "pmids")
        .withColumn("num_pmids", F.array_size(F.col("pmids")))
        .withColumnRenamed("curie", "id")
        .persist()
    )

    table = F.broadcast(curie_to_pmids)

    semmed_edges = (
        edges_df.alias("edges")
        .filter(F.col("primary_knowledge_source") == F.lit("infores:semmeddb"))
        # Enrich subject pubmed identifiers
        .join(
            table.alias("subj"),
            on=[F.col("edges.subject") == F.col("subj.id")],
            how="left",
        )
        # Enrich object pubmed identifiers
        .join(
            table.alias("obj"),
            on=[F.col("edges.object") == F.col("obj.id")],
            how="left",
        )
        .transform(compute_ngd)
        .withColumn("num_publications", F.size(F.col("publications")))
        # fmt: off
        .filter(
            # Retain only semmed edges more than 10 publications or ndg score above 0.6
            (F.col("num_publications") >= F.lit(publication_threshold)) & (F.col("ngd") > F.lit(ngd_threshold))
        )
        # fmt: on
        .select("edges.*")
    )

    edges_filtered = edges_df.filter(F.col("primary_knowledge_source") != F.lit("infores:semmeddb")).unionByName(
        semmed_edges
    )
    return edges_filtered


def compute_ngd(df: pyspark.sql.DataFrame, num_pairs: int = 3.7e7 * 20) -> pyspark.sql.DataFrame:
    """
    PySpark transformation to compute Normalized Google Distance (NGD).

    Args:
        df: Dataframe
        num_pairs: num_pairs
    Returns:
        Dataframe with ndg score
    """
    return (
        # Take first max_pmids elements from each array
        df.withColumn(
            "num_common_pmids", F.array_size(F.array_intersect(F.col("subj.pmids"), F.col("obj.pmids")))
        ).withColumn(
            "ngd",
            (
                F.greatest(F.log2(F.col("subj.num_pmids")), F.log2(F.col("obj.num_pmids")))
                - F.log2(F.col("num_common_pmids"))
            )
            / (F.log2(F.lit(num_pairs)) - F.least(F.log2(F.col("subj.num_pmids")), F.log2(F.col("obj.num_pmids")))),
        )
    )
