import pytest
from matrix.pipelines.integration import filters
from pandera.errors import SchemaError
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
        },
        {
            "name": "association",
            "children": [{"name": "chemical_to_chemical_association", "parent": "association"}],
        },
    ]


@pytest.fixture
def sample_edges(spark):
    return spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "CHEBI:002",
                "biolink:related_to",
            ),
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


@pytest.fixture
def sample_nodes(spark):
    # Note these are explicitly using PascalCase
    return spark.createDataFrame(
        [
            (
                "CHEBI:001",
                ["biolink:NamedThing", "biolink:Drug"],
            ),
            (
                "CHEBI:002",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
            ),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("all_categories", ArrayType(StringType()), False),
            ]
        ),
    )


def test_determine_most_specific_category(spark, sample_nodes):
    # When applying the biolink deduplicate
    result = filters.determine_most_specific_category(sample_nodes)
    expected = spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "biolink:Drug",
            ),
            (
                "CHEBI:002",
                "biolink:ChemicalEntity",
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


def test_determine_most_specific_category_unknown(spark):
    # When applying the biolink deduplicate

    nodes = spark.createDataFrame(
        [
            (
                "CHEBI:001",
                ["biolink:foo"],
            ),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("all_categories", ArrayType(StringType()), False),
            ]
        ),
    )

    with pytest.raises(SchemaError):
        filters.determine_most_specific_category(nodes)
