import pandas as pd
import pytest
from matrix.pipelines.integration import filters
from pyspark.sql.types import ArrayType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def sample_predicates():
    # These are explicitly using snake_case
    return [
        {
            "name": "related_to",
            "children": [
                {"name": "composed_primarily_of", "parent": "related_to"},
                {
                    "name": "related_to_at_concept_level",
                    "parent": "related_to",
                    "children": [
                        {"name": "broad_match", "parent": "related_to_at_concept_level"},
                    ],
                },
            ],
        }
    ]


@pytest.fixture
def sample_edges(spark):
    return spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "CHEBI:002",
                "biolink:RelatedTo",
            ),
            (
                "CHEBI:001",
                "CHEBI:002",
                "biolink:ComposedPrimarilyOf",
            ),
            (
                "CHEBI:001",
                "CHEBI:002",
                "biolink:RelatedToAtConceptLevel",
            ),
            (
                "CHEBI:002",
                "CHEBI:003",
                "biolink:RelatedToAtConceptLevel",
            ),
        ],
        schema=StructType(
            [
                StructField("subject", StringType(), False),
                StructField("object", StringType(), False),
                StructField("predicate", StringType(), False),
            ]
        ),
    )


@pytest.fixture
def sample_nodes(spark):
    # Note these are explicitly using PascalCase
    return spark.createDataFrame(
        [
            (
                "CHEBI:001",
                ["biolink:RelatedTo", "biolink:ComposedPrimarilyOf"],
            ),
            (
                "CHEBI:002",
                ["biolink:RelatedTo", "biolink:RelatedToAtConceptLevel", "biolink:BroadMatch"],
            ),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("all_categories", ArrayType(StringType()), False),
            ]
        ),
    )


def test_unnest(sample_predicates):
    # Given an input dictionary of hierarchical predicate definition

    # When calling the unnest function
    result = filters.unnest_biolink_hierarchy("predicate", sample_predicates, parents=[])
    expected = pd.DataFrame(
        [
            ["composed_primarily_of", ["related_to"]],
            ["broad_match", ["related_to", "related_to_at_concept_level"]],
            ["related_to_at_concept_level", ["related_to"]],
            ["related_to", []],
        ],
        columns=["predicate", "parents"],
    )

    # Then correct mapping returned
    assert result.equals(expected)


def test_biolink_deduplicate(spark, sample_edges, sample_predicates):
    # When applying the biolink deduplicate
    result = filters.biolink_deduplicate_edges(sample_edges, sample_predicates)
    expected = spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "CHEBI:002",
                "biolink:composed_primarily_of",
            ),
            (
                "CHEBI:001",
                "CHEBI:002",
                "biolink:related_to_at_concept_level",
            ),
            (
                "CHEBI:002",
                "CHEBI:003",
                "biolink:related_to_at_concept_level",
            ),
        ],
        schema=StructType(
            [
                StructField("subject", StringType(), False),
                StructField("object", StringType(), False),
                StructField("predicate", StringType(), False),
            ]
        ),
    )

    assertDataFrameEqual(result.select(*expected.columns), expected)


def test_determine_most_specific_category(spark, sample_nodes, sample_predicates):
    # When applying the biolink deduplicate
    result = filters.determine_most_specific_category(sample_nodes, sample_predicates)
    expected = spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "biolink:ComposedPrimarilyOf",
            ),
            (
                "CHEBI:002",
                "biolink:BroadMatch",
            ),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("category", StringType(), False),
            ]
        ),
    )

    assertDataFrameEqual(result.select(*expected.columns), expected)


def test_pascal_case():
    assert filters.to_pascal_case("related_to") == "RelatedTo"
    assert filters.to_pascal_case("related_to_at_concept_level") == "RelatedToAtConceptLevel"
    assert filters.to_pascal_case("composed_primarily_of") == "ComposedPrimarilyOf"
    assert filters.to_pascal_case("broad_match") == "BroadMatch"
    assert filters.to_pascal_case("named_thing") == "NamedThing"
    assert filters.to_pascal_case("entity") == "Entity"
