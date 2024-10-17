import pandas as pd
import pytest

from matrix.pipelines.integration import biolink

from pyspark.testing import assertDataFrameEqual
from pyspark.sql.types import ArrayType, StringType, StructField, StructType


@pytest.fixture
def sample_predicates():
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


def test_unnest(sample_predicates):
    # Given an input dictionary of hierarchical predicate definition

    # When calling the unnest function
    result = biolink.unnest(sample_predicates, parents=[])
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
    # Given sample edges and predicates
    predicates = spark.createDataFrame(biolink.unnest(sample_predicates))

    # When applying the biolink deduplicate
    result = biolink.biolink_deduplicate(sample_edges, predicates)
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

    # Then dataframe filtered correctly
    # NOTE: PySpark is a pain to compare?
    assert sorted(result.select("subject", "object", "predicate").collect()) == sorted(
        expected.select("subject", "object", "predicate").collect()
    )
