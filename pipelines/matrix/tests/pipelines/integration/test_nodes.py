from typing import Any, Dict

import pyspark.sql as ps
import pytest
from matrix.pipelines.integration import nodes
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


@pytest.fixture
def sample_nodes(spark):
    schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("name", StringType(), True),
            StructField("category", StringType(), False),
            StructField("description", StringType(), True),
            StructField("equivalent_identifiers", ArrayType(StringType()), True),
            StructField("all_categories", ArrayType(StringType()), True),
            StructField("publications", ArrayType(StringType()), True),
            StructField("labels", ArrayType(StringType()), True),
            StructField("international_resource_identifier", StringType(), True),
            StructField("upstream_data_source", ArrayType(StringType()), False),
        ]
    )
    data = [
        (
            "CHEBI:119157",
            "Drug1",
            "biolink:Drug",
            "Description1",
            ["CHEBI:119157"],
            ["biolink:Drug", "biolink:ChemicalEntity"],
            ["PMID:12345678"],
            ["Label1"],
            "http://example.com/1",
            ["source1"],
        ),
        (
            "MONDO:0005148",
            "Disease1",
            "biolink:Disease",
            "Description2",
            ["MONDO:0005148"],
            ["biolink:Disease"],
            ["PMID:23456789"],
            ["Label2"],
            "http://example.com/2",
            ["source2"],
        ),
        (
            "CHEBI:119157",
            "Drug1",
            "biolink:Drug",
            "Description3",
            ["CHEBI:119157"],
            ["biolink:Drug", "biolink:SmallMolecule"],
            ["PMID:34567890"],
            ["Label3"],
            "http://example.com/3",
            ["source3"],
        ),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_edges(spark):
    schema = StructType(
        [
            StructField("subject", StringType(), False),
            StructField("predicate", StringType(), False),
            StructField("object", StringType(), False),
            StructField("knowledge_level", StringType(), True),
            StructField("agent_type", StringType(), True),
            StructField("primary_knowledge_source", StringType(), True),
            StructField("aggregator_knowledge_source", ArrayType(StringType()), True),
            StructField("publications", ArrayType(StringType()), True),
            StructField("subject_aspect_qualifier", StringType(), True),
            StructField("subject_direction_qualifier", StringType(), True),
            StructField("object_aspect_qualifier", StringType(), True),
            StructField("object_direction_qualifier", StringType(), True),
            StructField("upstream_data_source", ArrayType(StringType()), False),
            StructField("num_references", IntegerType(), True),
            StructField("num_sentences", IntegerType(), True),
        ]
    )
    data = [
        (
            "CHEBI:119157",
            "biolink:treats",
            "MONDO:0005148",
            "knowledge_assertion",
            "manual_agent",
            "infores:semmeddb",
            ["infores:aggregator1"],
            ["PMID:12345678"],
            "aspect1",
            "increased",
            "aspect2",
            "decreased",
            ["source1"],
            10,
            10,
        ),
        (
            "CHEBI:120688",
            "biolink:interacts_with",
            "CHEBI:119157",
            "prediction",
            "computational_model",
            "infores:gtex",
            ["infores:aggregator2"],
            ["PMID:23456789"],
            "aspect3",
            "decreased",
            "aspect4",
            "increased",
            ["source2"],
            10,
            10,
        ),
        (
            "CHEBI:119157",
            "biolink:treats",
            "MONDO:0005148",
            "knowledge_assertion",
            "manual_agent",
            "infores:ubergraph",
            ["infores:aggregator3"],
            ["PMID:34567890"],
            "aspect5",
            "increased",
            "aspect6",
            "decreased",
            ["source3"],
            10,
            10,
        ),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_biolink_predicates():
    return [
        {
            "name": "related_to",
            "children": [
                {"name": "disease_has_location", "parent": "related_to"},
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
def nodenorm_response() -> Dict[str, Any]:
    return {
        "CHEMBL.COMPOUND:CHEMBL1201836": {
            "id": {
                "identifier": "CHEBI:28887",
                "label": "Ofatumumab",
                "description": "An ether in which the oxygen atom is connected to two methyl groups.",
            },
            "equivalent_identifiers": [
                {
                    "identifier": "CHEBI:28887",
                    "label": "dimethyl ether",
                    "description": "An ether in which the oxygen atom is connected to two methyl groups.",
                },
                {"identifier": "UNII:AM13FS69BX", "label": "DIMETHYL ETHER"},
            ],
            "type": ["biolink:SmallMolecule", "biolink:MolecularEntity"],
            "information_content": 92.8,
        }
    }


@pytest.fixture
def sample_biolink_category_hierarchy():
    """
    Returns a simplified subset of the biolink category hierarchy focusing on test-relevant categories.
    """
    return [
        {
            "name": "NamedThing",
            "children": [
                {
                    "name": "chemical_entity",
                    "parent": "NamedThing",
                    "children": [
                        {
                            "name": "molecular_mixture",
                            "parent": "chemical_entity",
                            "children": [{"name": "Drug", "parent": "molecular_mixture"}],
                        }
                    ],
                },
                {
                    "name": "biological_entity",
                    "parent": "NamedThing",
                    "children": [
                        {
                            "name": "disease_or_phenotypic_feature",
                            "parent": "biological_entity",
                            "children": [{"name": "Disease", "parent": "disease_or_phenotypic_feature"}],
                        }
                    ],
                },
                {
                    "name": "molecular_entity",
                    "parent": "NamedThing",
                    "children": [{"name": "SmallMolecule", "parent": "molecular_entity"}],
                },
            ],
        }
    ]


@pytest.mark.spark(
    help="This test relies on PYSPARK_PYTHON to be set appropriately, and sometimes does not work in VSCode"
)
def test_unify_nodes(spark, sample_nodes, sample_biolink_category_hierarchy):
    # Create two node datasets
    nodes1 = sample_nodes.filter(sample_nodes.id != "MONDO:0005148")
    nodes2 = sample_nodes.filter(sample_nodes.id != "CHEBI:119157")

    # Call the unify_nodes function
    result = nodes.union_and_deduplicate_nodes(sample_biolink_category_hierarchy, nodes1, nodes2)

    # Check the result
    assert isinstance(result, ps.DataFrame)
    assert result.count() == 2  # Should have deduplicated

    # Check if the properties are combined correctly for the duplicated node
    drug_node = result.filter(result.id == "CHEBI:119157").collect()[0]
    assert set(drug_node.all_categories) == {"biolink:Drug", "biolink:ChemicalEntity", "biolink:SmallMolecule"}
    assert set(drug_node.publications) == {"PMID:12345678", "PMID:34567890"}
    assert set(drug_node.upstream_data_source) == {"source1", "source3"}


@pytest.mark.spark(
    help="This test relies on PYSPARK_PYTHON to be set appropriately, and sometimes does not work in VSCode"
)
def test_correctly_identified_categories(spark, sample_nodes, sample_biolink_category_hierarchy):
    # Given: two node datasets
    nodes1 = sample_nodes
    nodes2 = sample_nodes.withColumn("category", F.lit("biolink:NamedThing"))

    # When: unifying the two datasets, putting nodes2 first -> meaning within each group, "first()" grabs the NamedThing
    result = nodes.union_and_deduplicate_nodes(sample_biolink_category_hierarchy, nodes1, nodes2)

    # Then: the most specific category is correctly identified
    assert result.filter(F.col("category") == "biolink:NamedThing").count() == 0


@pytest.mark.spark(
    help="This test relies on PYSPARK_PYTHON to be set appropriately, and sometimes does not work in VSCode"
)
def test_unify_edges(spark, sample_edges):
    # Create two edge datasets
    edges1 = sample_edges.filter(sample_edges.subject != "CHEBI:120688")
    edges2 = sample_edges.filter(sample_edges.subject != "CHEBI:119157")

    # Call the unify_edges function
    result = nodes.union_edges(edges1, edges2)

    # Check the result
    assert isinstance(result, ps.DataFrame)
    assert result.count() == 2  # Should have deduplicated

    # Check if the properties are combined correctly for the duplicated edge
    treat_edge = result.filter((result.subject == "CHEBI:119157") & (result.object == "MONDO:0005148")).collect()[0]
    assert treat_edge.knowledge_level == "knowledge_assertion"
    assert treat_edge.agent_type == "manual_agent"
    assert treat_edge.primary_knowledge_source == "infores:semmeddb"
    assert set(treat_edge.aggregator_knowledge_source) == {"infores:aggregator1", "infores:aggregator3"}
    assert set(treat_edge.publications) == {"PMID:12345678", "PMID:34567890"}
    assert set(treat_edge.upstream_data_source) == {"source1", "source3"}
