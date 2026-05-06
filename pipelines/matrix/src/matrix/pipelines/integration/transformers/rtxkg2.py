import logging
from typing import Dict

import pyspark.sql as ps
import pyspark.sql.functions as f
import pyspark.sql.types as T

from .transformer import GraphTransformer

logger = logging.getLogger(__name__)


RTX_SEPARATOR = r"\|"


class RTXTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform RTX KG2 nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        match self._version:
            case "v2.7.3":
                df = transform_nodes_v2_7_3(nodes_df)
            case "v2.10.0_validated":
                df = transform_nodes_v2_10_0_validated(nodes_df)
            case _:
                raise NotImplementedError(f"No nodes transformer code implemented for version: {self._version}")
        return df

    def transform_edges(
        self, edges_df: ps.DataFrame, curie_to_pmids: ps.DataFrame, semmed_filters: Dict[str, str], **kwargs
    ) -> ps.DataFrame:
        """Transform RTX KG2 edges to our target schema.

        Args:
            edges_df: Edges DataFrame.
        Returns:
            Transformed DataFrame.
        """
        match self._version:
            case "v2.7.3":
                df = transform_edges_v2_7_3(edges_df, curie_to_pmids, semmed_filters)
            case "v2.10.0_validated":
                df = transform_edges_v2_10_0_validated(edges_df, curie_to_pmids, semmed_filters)
            case _:
                raise NotImplementedError(f"No edges transformer code implemented for version: {self._version}")

        return df


def transform_nodes_v2_7_3(nodes_df: ps.DataFrame):
    # fmt: off
    df = (nodes_df
          .withColumn("upstream_data_source",              f.array(f.lit("rtxkg2")))
          .withColumn("labels",                            f.split(f.col(":LABEL"), RTX_SEPARATOR))
          .withColumn("all_categories",                    f.split(f.col("all_categories:string[]"), RTX_SEPARATOR))
          .withColumn("equivalent_identifiers",            f.split(f.col("equivalent_curies:string[]"), RTX_SEPARATOR))
          .withColumn("publications",                      f.split(f.col("publications:string[]"), RTX_SEPARATOR).cast(T.ArrayType(T.StringType())))
          .withColumn("international_resource_identifier", f.col("iri"))
          .withColumnRenamed("id:ID", "id")
    )
    # fmt: on
    return df


def transform_nodes_v2_10_0_validated(nodes_df: ps.DataFrame):
    # fmt: off
    df = (
        nodes_df
        .withColumn("upstream_data_source",              f.array(f.lit("rtxkg2")))
        .withColumn("labels",                            f.lit(None).cast(T.ArrayType(T.StringType()))) # TODO: replace this with _LABEL once updated rtx-kg2 is available
        .withColumn("all_categories",                    f.split(f.col("all_categories"), RTX_SEPARATOR))
        .withColumn("equivalent_identifiers",            f.split(f.col("equivalent_curies"), RTX_SEPARATOR))
        .withColumn("publications",                      f.split(f.col("publications"), RTX_SEPARATOR).cast(T.ArrayType(T.StringType())))
        .withColumn("international_resource_identifier", f.col("iri"))
    )
    # fmt: on
    return df


def transform_edges_v2_7_3(edges_df: ps.DataFrame, curie_to_pmids: ps.DataFrame, semmed_filters: Dict[str, str]):
    # fmt: off
    df = (edges_df
          # Qualifiers — not present in v2.7.3
          .withColumn("qualified_predicate",          f.lit(None).cast(T.StringType()))
          .withColumn("subject_aspect_qualifier",      f.lit(None).cast(T.StringType()))
          .withColumn("subject_direction_qualifier",   f.lit(None).cast(T.StringType()))
          .withColumn("subject_part_qualifier",        f.lit(None).cast(T.StringType()))
          .withColumn("object_aspect_qualifier",       f.lit(None).cast(T.StringType()))
          .withColumn("object_direction_qualifier",    f.lit(None).cast(T.StringType()))
          .withColumn("object_specialization_qualifier", f.lit(None).cast(T.StringType()))
          .withColumn("object_part_qualifier",         f.lit(None).cast(T.StringType()))
          .withColumn("species_context_qualifier",     f.lit(None).cast(T.StringType()))
          .withColumn("disease_context_qualifier",     f.lit(None).cast(T.StringType()))
          .withColumn("frequency_qualifier",           f.lit(None).cast(T.StringType()))
          .withColumn("qualifiers",                    f.lit(None).cast(T.StringType()))
          .withColumn("stage_qualifier",               f.lit(None).cast(T.StringType()))
          .withColumn("anatomical_context_qualifier",  f.lit(None).cast(T.StringType()))
          .withColumn("onset_qualifier",               f.lit(None).cast(T.StringType()))
          .withColumn("sex_qualifier",                 f.lit(None).cast(T.StringType()))
          # Provenance — v2.7.3 uses annotated column names
          .withColumn("aggregator_knowledge_source",   f.split(f.col("knowledge_source:string[]"), RTX_SEPARATOR))
          .withColumn("publications",                  f.split(f.col("publications:string[]"), RTX_SEPARATOR))
          .withColumn("upstream_data_source",          f.array(f.lit("rtxkg2")))
          .withColumn("knowledge_level",               f.lit(None).cast(T.StringType()))
          .withColumn("primary_knowledge_source",      f.col("aggregator_knowledge_source").getItem(0))
          # Quantitative attributes
          .withColumn("has_confidence_score",          f.lit(None).cast(T.FloatType()))
          .withColumn("extraction_confidence_score",   f.lit(None).cast(T.FloatType()))
          .withColumn("affinity",                      f.lit(None).cast(T.FloatType()))
          .withColumn("affinity_parameter",            f.lit(None).cast(T.StringType()))
          .withColumn("supporting_study_method_type",  f.lit(None).cast(T.StringType()))
          .transform(filter_semmed, curie_to_pmids, **semmed_filters)
    )
    # fmt: on
    return df


def transform_edges_v2_10_0_validated(
    edges_df: ps.DataFrame, curie_to_pmids: ps.DataFrame, semmed_filters: Dict[str, str]
):
    # fmt: off
    df = (edges_df
            # Qualifiers — v2.10 exposes object_aspect_qualifier and object_direction_qualifier directly
            .withColumn("qualified_predicate",           f.lit(None).cast(T.StringType()))
            .withColumn("subject_aspect_qualifier",      f.lit(None).cast(T.StringType()))
            .withColumn("subject_direction_qualifier",   f.lit(None).cast(T.StringType()))
            .withColumn("subject_part_qualifier",        f.lit(None).cast(T.StringType()))
            .withColumn("object_aspect_qualifier",       f.col("object_aspect_qualifier").cast(T.StringType()))
            .withColumn("object_direction_qualifier",    f.col("object_direction_qualifier").cast(T.StringType()))
            .withColumn("object_specialization_qualifier", f.lit(None).cast(T.StringType()))
            .withColumn("object_part_qualifier",         f.lit(None).cast(T.StringType()))
            .withColumn("species_context_qualifier",     f.lit(None).cast(T.StringType()))
            .withColumn("disease_context_qualifier",     f.lit(None).cast(T.StringType()))
            .withColumn("frequency_qualifier",           f.lit(None).cast(T.StringType()))
            .withColumn("qualifiers",                    f.lit(None).cast(T.StringType()))
            .withColumn("stage_qualifier",               f.lit(None).cast(T.StringType()))
            .withColumn("anatomical_context_qualifier",  f.lit(None).cast(T.StringType()))
            .withColumn("onset_qualifier",               f.lit(None).cast(T.StringType()))
            .withColumn("sex_qualifier",                 f.lit(None).cast(T.StringType()))
            # Provenance
            .withColumn("aggregator_knowledge_source",   f.split(f.col("aggregator_knowledge_source"), RTX_SEPARATOR))
            .withColumn("publications",                  f.split(f.col("publications"), RTX_SEPARATOR))
            .withColumn("upstream_data_source",          f.array(f.lit("rtxkg2")))
            # Quantitative attributes
            .withColumn("num_references",                f.lit(None).cast(T.IntegerType()))
            .withColumn("num_sentences",                 f.lit(None).cast(T.IntegerType()))
            .withColumn("has_confidence_score",          f.lit(None).cast(T.FloatType()))
            .withColumn("extraction_confidence_score",   f.lit(None).cast(T.FloatType()))
            .withColumn("affinity",                      f.lit(None).cast(T.FloatType()))
            .withColumn("affinity_parameter",            f.lit(None).cast(T.StringType()))
            .withColumn("supporting_study_method_type",  f.lit(None).cast(T.StringType()))
            .transform(filter_semmed, curie_to_pmids, **semmed_filters)
    )
    # fmt: on
    return df


def filter_semmed(
    edges_df: ps.DataFrame,
    curie_to_pmids: ps.DataFrame,
    publication_threshold: int,
    ngd_threshold: float,
    limit_pmids: int,
) -> ps.DataFrame:
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

    sorted_pmids = f.sort_array(f.from_json("pmids", T.ArrayType(T.IntegerType())))
    curie_to_pmids = (
        curie_to_pmids.withColumn("pmids", f.slice(sorted_pmids, 1, limit_pmids))
        .withColumn("num_pmids", f.array_size("pmids"))
        .withColumnRenamed("curie", "id")
        .cache()
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            f'{{"curie_to_pmids shape": "{curie_to_pmids.count():_}x{len(curie_to_pmids.columns)}", "curie_to_pmids schema": "{curie_to_pmids.schema.simpleString()}"}}',
        )
        logger.info(
            f'{{"edges_df shape": "{edges_df.count():_}x{len(edges_df.columns)}", "edges_df schema": "{edges_df.schema.simpleString()}"}}',
        )

    semmeddb_is_only_knowledge_source = (f.size("aggregator_knowledge_source") == 1) & (
        f.col("aggregator_knowledge_source").getItem(0) == "infores:semmeddb"
    )
    table = f.broadcast(curie_to_pmids)
    single_semmed_edges = (
        edges_df.filter(semmeddb_is_only_knowledge_source)
        .alias("edges")
        .join(
            table.alias("subj"),
            on=[f.col("edges.subject") == f.col("subj.id")],
            how="left",
        )
        .join(
            table.alias("obj"),
            on=[f.col("edges.object") == f.col("obj.id")],
            how="left",
        )
        .transform(compute_ngd)
        .filter(
            # Retain only semmed edges more than 10 publications or ndg score below/equal 0.6
            (f.size("publications") >= f.lit(publication_threshold)) & (f.col("ngd") <= f.lit(ngd_threshold))
        )
        .select("edges.*")
    )
    edges_filtered = edges_df.filter(~semmeddb_is_only_knowledge_source).unionByName(single_semmed_edges)
    edges_filtered.explain()
    return edges_filtered.withColumn(
        "num_references",
        f.when(
            f.col("publications").isNotNull(),
            f.size(f.array_distinct(f.col("publications"))),
        ).cast(T.IntegerType()),
    )


def compute_ngd(df: ps.DataFrame, num_pairs: int = 3.7e7 * 20) -> ps.DataFrame:
    """
    PySpark transformation to compute Normalized Google Distance (NGD).

    Args:
        df: Dataframe
        num_pairs: num_pairs
    Returns:
        Dataframe with ndg score
    """

    nbr_common_pmids = f.array_size(f.array_intersect("subj.pmids", "obj.pmids"))
    num_pmids = ("subj.num_pmids", "obj.num_pmids")

    return df.withColumn(
        "ngd",
        (f.log2(f.greatest(*num_pmids)) - f.log2(nbr_common_pmids))
        / (f.log2(f.lit(num_pairs)) - f.log2(f.least(*num_pmids))),
    )
