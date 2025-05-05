import pandas as pd
import pytest
from matrix.pipelines.filtering import filters
from pandera.errors import SchemaError
from pyspark.sql.types import ArrayType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def sample_nodes(spark):
    return spark.createDataFrame(
        [
            (
                "CHEBI:001",
                ["biolink:NamedThing", "biolink:Drug"],
                ["rtxkg2", "robokop"],
            ),
            (
                "CHEBI:002",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                ["rtxkg2", "robokop"],
            ),
            (
                "CHEBI:003",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                ["robokop"],
            ),
            (
                "CHEBI:004",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                ["rtxkg2"],
            ),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("all_categories", ArrayType(StringType()), False),
                StructField("upstream_data_source", ArrayType(StringType()), False),
            ]
        ),
    )


@pytest.fixture
def sample_edges(spark):
    return spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "CHEBI:002",
                "biolink:related_to",
                ["rtxkg2", "robokop"],
            ),
            (
                "CHEBI:002",
                "CHEBI:003",
                "biolink:subclass_of",
                ["robokop"],
            ),
            (
                "CHEBI:004",
                "CHEBI:001",
                "biolink:similar_to",
                ["rtxkg2"],
            ),
            (
                "CHEBI:004",
                "CHEBI:001",
                "biolink:chemically_similar_to",
                ["rtxkg2"],
            ),
        ],
        schema=StructType(
            [
                StructField("subject", StringType(), False),
                StructField("object", StringType(), False),
                StructField("predicate", StringType(), False),
                StructField("kg_sources", ArrayType(StringType()), False),
            ]
        ),
    )


def test_source_filter_nodes(spark, sample_nodes):
    result = filters.keep_rows_containing(
        input_df=sample_nodes,
        column="upstream_data_source",
        keep_list=["rtxkg2"],
    )
    expected = spark.createDataFrame(
        [
            (
                "CHEBI:001",
                ["biolink:NamedThing", "biolink:Drug"],
                ["rtxkg2", "robokop"],
            ),
            (
                "CHEBI:002",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                ["rtxkg2", "robokop"],
            ),
            (
                "CHEBI:004",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                ["rtxkg2"],
            ),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("all_categories", ArrayType(StringType()), False),
                StructField("upstream_data_source", ArrayType(StringType()), False),
            ]
        ),
    )
    assertDataFrameEqual(result, expected)


def test_source_filter_edges(spark, sample_edges):
    result = filters.keep_rows_containing(
        input_df=sample_edges,
        column="kg_sources",
        keep_list=["rtxkg2"],
    )
    expected = spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "CHEBI:002",
                "biolink:related_to",
                ["rtxkg2", "robokop"],
            ),
            (
                "CHEBI:004",
                "CHEBI:001",
                "biolink:similar_to",
                ["rtxkg2"],
            ),
            (
                "CHEBI:004",
                "CHEBI:001",
                "biolink:chemically_similar_to",
                ["rtxkg2"],
            ),
        ],
        schema=StructType(
            [
                StructField("subject", StringType(), False),
                StructField("object", StringType(), False),
                StructField("predicate", StringType(), False),
                StructField("kg_sources", ArrayType(StringType()), False),
            ]
        ),
    )
    assertDataFrameEqual(result, expected)


def test_biolink_deduplicate(spark, sample_edges):
    result = filters.biolink_deduplicate_edges(sample_edges)
    expected = spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "CHEBI:002",
                "biolink:related_to",
                ["rtxkg2", "robokop"],
            ),
            (
                "CHEBI:002",
                "CHEBI:003",
                "biolink:subclass_of",
                ["robokop"],
            ),
            (
                "CHEBI:004",
                "CHEBI:001",
                "biolink:chemically_similar_to",
                ["rtxkg2"],
            ),
        ],
        schema=StructType(
            [
                StructField("subject", StringType(), False),
                StructField("object", StringType(), False),
                StructField("predicate", StringType(), False),
                StructField("kg_sources", ArrayType(StringType()), False),
            ]
        ),
    )
    assertDataFrameEqual(result.select(*expected.columns), expected)


def test_filter_triples(spark):
    test_edges = spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "biolink:Drug",
                "biolink:physically_interacts_with",
                "CHEBI:002",
                "biolink:Drug",
            ),
            (
                "CHEBI:001",
                "biolink:Drug",
                "biolink:treats",
                "UMLS:001",
                "biolink:Disease",
            ),
        ],
        schema=StructType(
            [
                StructField("subject", StringType(), False),
                StructField("subject_category", StringType(), False),
                StructField("predicate", StringType(), False),
                StructField("object", StringType(), False),
                StructField("object_category", StringType(), False),
            ]
        ),
    )
    result = filters.filter_triples(
        edges_df=test_edges,
        triples_to_exclude=[["biolink:Drug", "biolink:physically_interacts_with", "biolink:Drug"]],
    )
    expected = spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "biolink:Drug",
                "biolink:treats",
                "UMLS:001",
                "biolink:Disease",
            ),
        ],
        schema=StructType(
            [
                StructField("subject", StringType(), False),
                StructField("subject_category", StringType(), False),
                StructField("predicate", StringType(), False),
                StructField("object", StringType(), False),
                StructField("object_category", StringType(), False),
            ]
        ),
    )
    assertDataFrameEqual(result, expected)


def test_remove_rows_by_column(spark):
    test_nodes = spark.createDataFrame(
        [
            ("CHEBI:001", "biolink:Drug"),
            ("OT:001", "biolink:OrganismTaxon"),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("category", StringType(), False),
            ]
        ),
    )
    result = filters.remove_rows_by_column(
        input_df=test_nodes,
        column="category",
        remove_list=["biolink:OrganismTaxon"],
    )
    expected = spark.createDataFrame(
        [
            ("CHEBI:001", "biolink:Drug"),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("category", StringType(), False),
            ]
        ),
    )
    assertDataFrameEqual(result, expected)
