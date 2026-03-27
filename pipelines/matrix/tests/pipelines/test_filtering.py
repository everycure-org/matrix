import json

import pytest
from matrix.pipelines.filtering import filters
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def sample_nodes(spark):
    return spark.createDataFrame(
        [
            (
                "CHEBI:001",
                ["biolink:NamedThing", "biolink:Drug"],
                "biolink:Drug",
                ["rtxkg2", "robokop"],
            ),
            (
                "CHEBI:002",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                "biolink:ChemicalEntity",
                ["rtxkg2", "robokop"],
            ),
            (
                "CHEBI:003",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                "biolink:ChemicalEntity",
                ["robokop"],
            ),
            (
                "CHEBI:004",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                "biolink:ChemicalEntity",
                ["rtxkg2"],
            ),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("all_categories", ArrayType(StringType()), False),
                StructField("category", StringType(), False),
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
                "biolink:Drug",
                ["rtxkg2", "robokop"],
            ),
            (
                "CHEBI:002",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                "biolink:ChemicalEntity",
                ["rtxkg2", "robokop"],
            ),
            (
                "CHEBI:004",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                "biolink:ChemicalEntity",
                ["rtxkg2"],
            ),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("all_categories", ArrayType(StringType()), False),
                StructField("category", StringType(), False),
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
            (
                "CHEBI:002",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                "biolink:ChemicalEntity",
                ["rtxkg2", "robokop"],
            ),
            ("CHEBI:003", ["biolink:NamedThing", "biolink:ChemicalEntity"], "biolink:ChemicalEntity", ["robokop"]),
            ("CHEBI:004", ["biolink:NamedThing", "biolink:ChemicalEntity"], "biolink:ChemicalEntity", ["rtxkg2"]),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("all_categories", ArrayType(StringType()), False),
                StructField("category", StringType(), False),
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
            ("CHEBI:001", ["biolink:NamedThing", "biolink:Drug"], "biolink:Drug", ["rtxkg2", "robokop"]),
            (
                "CHEBI:002",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
                "biolink:ChemicalEntity",
                ["rtxkg2", "robokop"],
            ),
            ("CHEBI:003", ["biolink:NamedThing", "biolink:ChemicalEntity"], "biolink:ChemicalEntity", ["robokop"]),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("all_categories", ArrayType(StringType()), False),
                StructField("category", StringType(), False),
                StructField("upstream_data_source", ArrayType(StringType()), False),
            ]
        ),
    )
    assertDataFrameEqual(result, expected)


def test_deduplicate_edges(spark):
    """Test DeduplicateEdges collapses same-SPO edges and preserves per-source detail.

    Given 3 edges with the same (subject, predicate, object) but different
    primary_knowledge_source, object_direction_qualifier, knowledge_level, and publications
    When we apply DeduplicateEdges
    Then the output should have 1 row with:
    - primary_knowledge_sources containing all 3 PKS values
    - publications as the flat union of all 3 publication lists
    - knowledge_level absent as a top-level column (moved into source_edge_properties)
    - source_edge_properties containing one entry per PKS with the correct qualifier values
    """
    schema = StructType(
        [
            StructField("subject", StringType(), False),
            StructField("predicate", StringType(), False),
            StructField("object", StringType(), False),
            StructField("qualified_predicate", StringType(), True),
            StructField("subject_aspect_qualifier", StringType(), True),
            StructField("subject_direction_qualifier", StringType(), True),
            StructField("subject_part_qualifier", StringType(), True),
            StructField("object_aspect_qualifier", StringType(), True),
            StructField("object_direction_qualifier", StringType(), True),
            StructField("object_specialization_qualifier", StringType(), True),
            StructField("object_part_qualifier", StringType(), True),
            StructField("species_context_qualifier", StringType(), True),
            StructField("disease_context_qualifier", StringType(), True),
            StructField("frequency_qualifier", StringType(), True),
            StructField("qualifiers", StringType(), True),
            StructField("stage_qualifier", StringType(), True),
            StructField("anatomical_context_qualifier", StringType(), True),
            StructField("onset_qualifier", StringType(), True),
            StructField("sex_qualifier", StringType(), True),
            StructField("knowledge_level", StringType(), True),
            StructField("agent_type", StringType(), True),
            StructField("primary_knowledge_source", StringType(), True),
            StructField("primary_knowledge_sources", ArrayType(StringType()), True),
            StructField("aggregator_knowledge_source", ArrayType(StringType()), True),
            StructField("publications", ArrayType(StringType()), True),
            StructField("upstream_data_source", ArrayType(StringType()), True),
            StructField("num_references", IntegerType(), True),
            StructField("num_sentences", IntegerType(), True),
            StructField("has_confidence_score", FloatType(), True),
            StructField("extraction_confidence_score", FloatType(), True),
            StructField("affinity", FloatType(), True),
            StructField("affinity_parameter", StringType(), True),
            StructField("supporting_study_method_type", StringType(), True),
        ]
    )
    input_df = spark.createDataFrame(
        [
            (
                "CHEBI:1",
                "biolink:regulates",
                "MONDO:1",
                None,
                None,
                None,
                None,
                None,
                "increased",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "not_provided",
                "manual_agent",
                "infores:semmeddb",
                ["infores:semmeddb", "infores:ctd", "infores:monarch"],
                None,
                ["PMID:1"],
                ["semmeddb"],
                5,
                2,
                None,
                None,
                None,
                None,
                None,
            ),
            (
                "CHEBI:1",
                "biolink:regulates",
                "MONDO:1",
                None,
                None,
                None,
                None,
                None,
                "decreased",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "knowledge_assertion",
                "manual_agent",
                "infores:ctd",
                ["infores:semmeddb", "infores:ctd", "infores:monarch"],
                None,
                ["PMID:2"],
                ["ctd"],
                3,
                1,
                None,
                None,
                None,
                None,
                None,
            ),
            (
                "CHEBI:1",
                "biolink:regulates",
                "MONDO:1",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "not_provided",
                "automated_agent",
                "infores:monarch",
                ["infores:semmeddb", "infores:ctd", "infores:monarch"],
                None,
                ["PMID:3"],
                ["monarch"],
                1,
                0,
                None,
                None,
                None,
                None,
                None,
            ),
        ],
        schema=schema,
    )

    result = filters.DeduplicateEdges().apply(input_df)

    # All 3 edges collapse to 1 row
    assert result.count() == 1

    # knowledge_level is not a top-level column (moved to source_edge_properties)
    assert "knowledge_level" not in result.columns

    row = result.collect()[0]

    # Cross-edge provenance contains all 3 sources
    assert set(row["primary_knowledge_sources"]) == {"infores:semmeddb", "infores:ctd", "infores:monarch"}

    # publications is the flat union of all 3 lists
    assert set(row["publications"]) == {"PMID:1", "PMID:2", "PMID:3"}

    # num_references is the sum across all 3 edges
    assert row["num_references"] == 9

    # source_edge_properties is a JSON string keyed by primary_knowledge_source.
    # Scalar qualifier fields are arrays of all distinct non-null observed values
    # (collect_set preserves all unique values and ignores nulls).
    props = json.loads(row["source_edge_properties"])
    assert set(props.keys()) == {"infores:semmeddb", "infores:ctd", "infores:monarch"}
    assert props["infores:semmeddb"]["object_direction_qualifier"] == ["increased"]
    assert props["infores:ctd"]["object_direction_qualifier"] == ["decreased"]
    assert props["infores:monarch"]["object_direction_qualifier"] == []
    assert props["infores:semmeddb"]["knowledge_level"] == ["not_provided"]
    assert props["infores:ctd"]["knowledge_level"] == ["knowledge_assertion"]
