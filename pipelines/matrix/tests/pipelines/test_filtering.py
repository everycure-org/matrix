import pytest
from matrix.pipelines.filtering import filters
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
    """Test KeepRowsContaining on nodes DataFrame.

    Given a DataFrame with nodes containing upstream data sources
    When we apply KeepRowsContaining to keep only nodes from specific sources
    Then only nodes from those sources should remain
    """
    result = filters.KeepRowsContaining(column="upstream_data_source", keep_list=["rtxkg2"]).apply(sample_nodes)

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
    """Test KeepRowsContaining on edges DataFrame.

    Given a DataFrame with edges containing knowledge graph sources
    When we apply KeepRowsContaining to keep only edges from rtxkg2
    Then only edges from rtxkg2 should remain
    """
    result = filters.KeepRowsContaining(column="kg_sources", keep_list=["rtxkg2"]).apply(sample_edges)

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
    """Test BiolinkDeduplicateEdges functionality.

    Given a DataFrame with edges that may have redundant biolink predicates
    When we apply BiolinkDeduplicateEdges
    Then redundant edges should be removed while keeping the most specific predicates
    """
    result = filters.BiolinkDeduplicateEdges().apply(sample_edges)
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


def test_remove_rows_by_column_filter(spark):
    """Test RemoveRowsByColumn functionality.

    Given a DataFrame with nodes containing specific categories
    When we apply RemoveRowsByColumn to remove those categories
    Then only nodes with the remaining categories should remain
    """
    test_nodes = spark.createDataFrame(
        [
            ("CHEBI:001", "biolink:Drug"),
            ("OT:001", "biolink:OrganismTaxon"),
            ("CHEBI:002", "biolink:ChemicalEntity"),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("category", StringType(), False),
            ]
        ),
    )

    result = filters.RemoveRowsByColumn(
        column="category", remove_list=["biolink:OrganismTaxon", "biolink:ChemicalEntity"]
    ).apply(test_nodes)

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


def test_triple_pattern_filter(spark):
    """Test TriplePattern functionality.

    Given a DataFrame with edges containing subject-predicate-object patterns
    When we apply TriplePattern to exclude specific patterns
    Then only edges not matching the excluded patterns should remain
    """
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

    result = filters.TriplePattern(
        triples_to_exclude=[["biolink:Drug", "biolink:physically_interacts_with", "biolink:Drug"]]
    ).apply(test_edges)

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


def test_triple_pattern_filter_multiple_patterns(spark):
    """Test TriplePattern with multiple patterns to exclude.

    Given a DataFrame with edges containing various subject-predicate-object patterns
    When we apply TriplePattern to exclude multiple patterns
    Then only edges not matching any of the excluded patterns should remain
    """
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
            (
                "CHEBI:003",
                "biolink:Drug",
                "biolink:chemically_similar_to",
                "CHEBI:004",
                "biolink:Drug",
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

    result = filters.TriplePattern(
        triples_to_exclude=[
            ["biolink:Drug", "biolink:physically_interacts_with", "biolink:Drug"],
            ["biolink:Drug", "biolink:chemically_similar_to", "biolink:Drug"],
        ]
    ).apply(test_edges)

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


def test_remove_rows_by_column_overlap_without_excluded_sources(spark, sample_nodes):
    filter = filters.RemoveRowsByColumnOverlap(
        column="all_categories", remove_list=["biolink:Drug"], excluded_sources=[]
    )
    result = filter.apply(sample_nodes)
    expected = spark.createDataFrame(
        [
            ("CHEBI:002", ["biolink:NamedThing", "biolink:ChemicalEntity"], ["rtxkg2", "robokop"]),
            ("CHEBI:003", ["biolink:NamedThing", "biolink:ChemicalEntity"], ["robokop"]),
            ("CHEBI:004", ["biolink:NamedThing", "biolink:ChemicalEntity"], ["rtxkg2"]),
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


def test_remove_rows_by_column_overlap_with_excluded_sources(spark, sample_nodes):
    filter = filters.RemoveRowsByColumnOverlap(
        column="all_categories", remove_list=["biolink:ChemicalEntity"], excluded_sources=["robokop"]
    )
    result = filter.apply(sample_nodes)
    expected = spark.createDataFrame(
        [
            ("CHEBI:001", ["biolink:NamedThing", "biolink:Drug"], ["rtxkg2", "robokop"]),
            ("CHEBI:002", ["biolink:NamedThing", "biolink:ChemicalEntity"], ["rtxkg2", "robokop"]),
            ("CHEBI:003", ["biolink:NamedThing", "biolink:ChemicalEntity"], ["robokop"]),
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
