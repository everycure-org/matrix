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
                "biolink:NamedThing",
                ["biolink:NamedThing", "biolink:Drug"],
            ),
            (
                "CHEBI:002",
                "biolink:NamedThing",
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
            ),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("category", StringType(), True),
                StructField("all_categories", ArrayType(StringType()), False),
            ]
        ),
    )


def test_determine_most_specific_category_core_drug_rule(spark):
    """Test Rule 1: Core drugs always get biolink:Drug"""
    nodes = spark.createDataFrame(
        [("DRUG:001", "biolink:ChemicalEntity", "drug", ["biolink:ChemicalEntity", "biolink:NamedThing"])],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("category", StringType(), True),
                StructField("core_type", StringType(), True),
                StructField("all_categories", ArrayType(StringType()), False),
            ]
        ),
    )

    result = filters.determine_most_specific_category(nodes)
    assert result.collect()[0].category == "biolink:Drug"


def test_determine_most_specific_category_core_disease_rule(spark):
    """Test Rule 2: Core diseases always get biolink:Disease"""
    nodes = spark.createDataFrame(
        [("DISEASE:001", "biolink:Phenotype", "disease", ["biolink:Phenotype", "biolink:NamedThing"])],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("category", StringType(), True),
                StructField("core_type", StringType(), True),
                StructField("all_categories", ArrayType(StringType()), False),
            ]
        ),
    )

    result = filters.determine_most_specific_category(nodes)
    assert result.collect()[0].category == "biolink:Disease"


def test_determine_most_specific_category_namedthing_fallback(spark):
    """Test Rule 3: NamedThing-only nodes preserve valid category"""
    nodes = spark.createDataFrame(
        [
            ("CHEBI:001", "biolink:ChemicalEntity", ["biolink:NamedThing"])  # Only NamedThing in all_categories
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("category", StringType(), True),
                StructField("all_categories", ArrayType(StringType()), False),
            ]
        ),
    )

    result = filters.determine_most_specific_category(nodes)
    # Should preserve the more specific category from 'category' column
    assert result.collect()[0].category == "biolink:ChemicalEntity"


def test_determine_most_specific_category_normal_hierarchy(spark):
    """Test Rule 4: Normal most-specific logic"""
    nodes = spark.createDataFrame(
        [("CHEBI:002", "biolink:NamedThing", ["biolink:Drug", "biolink:MolecularMixture", "biolink:NamedThing"])],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("category", StringType(), True),
                StructField("all_categories", ArrayType(StringType()), False),
            ]
        ),
    )

    result = filters.determine_most_specific_category(nodes)
    # Should pick most specific from all_categories
    assert result.collect()[0].category == "biolink:Drug"


def test_determine_most_specific_category(spark, sample_nodes):
    # When applying the biolink deduplicate
    result = filters.determine_most_specific_category(sample_nodes)
    expected = spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "biolink:Drug",  # Most specific from ["biolink:NamedThing", "biolink:Drug"]
                ["biolink:NamedThing", "biolink:Drug"],
            ),
            (
                "CHEBI:002",
                "biolink:ChemicalEntity",  # Most specific from ["biolink:NamedThing", "biolink:ChemicalEntity"]
                ["biolink:NamedThing", "biolink:ChemicalEntity"],
            ),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("category", StringType(), True),
                StructField("all_categories", ArrayType(StringType()), False),
            ]
        ),
    )

    assertDataFrameEqual(result.select(*expected.columns), expected)


def test_determine_most_specific_category_unknown(spark):
    """Test handling of unknown categories"""
    nodes = spark.createDataFrame(
        [
            (
                "CHEBI:001",
                "biolink:NamedThing",
                ["biolink:foo"],  # Invalid category
            ),
        ],
        schema=StructType(
            [
                StructField("id", StringType(), False),
                StructField("category", StringType(), True),
                StructField("all_categories", ArrayType(StringType()), False),
            ]
        ),
    )
    result = filters.determine_most_specific_category(nodes)
    # Since "biolink:foo" is invalid, should fallback to original category
    assert result.collect()[0].category == "biolink:NamedThing"
