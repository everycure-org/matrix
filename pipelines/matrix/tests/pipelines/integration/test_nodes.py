import pytest
from matrix.pipelines.integration import nodes
from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual


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
            ["biolink:Drug", "biolink:ChemicalSubstance"],
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
            StructField("primary_knowledge_source", StringType(), True),
            StructField("aggregator_knowledge_source", ArrayType(StringType()), True),
            StructField("publications", ArrayType(StringType()), True),
            StructField("subject_aspect_qualifier", StringType(), True),
            StructField("subject_direction_qualifier", StringType(), True),
            StructField("object_aspect_qualifier", StringType(), True),
            StructField("object_direction_qualifier", StringType(), True),
            StructField("upstream_data_source", ArrayType(StringType()), False),
        ]
    )
    data = [
        (
            "CHEBI:119157",
            "biolink:treats",
            "MONDO:0005148",
            "knowledge_assertion",
            "infores:semmeddb",
            ["infores:aggregator1"],
            ["PMID:12345678"],
            "aspect1",
            "increased",
            "aspect2",
            "decreased",
            ["source1"],
        ),
        (
            "CHEBI:120688",
            "biolink:interacts_with",
            "CHEBI:119157",
            "prediction",
            "infores:gtex",
            ["infores:aggregator2"],
            ["PMID:23456789"],
            "aspect3",
            "decreased",
            "aspect4",
            "increased",
            ["source2"],
        ),
        (
            "CHEBI:119157",
            "biolink:treats",
            "MONDO:0005148",
            "knowledge_assertion",
            "infores:ubergraph",
            ["infores:aggregator3"],
            ["PMID:34567890"],
            "aspect5",
            "increased",
            "aspect6",
            "decreased",
            ["source3"],
        ),
    ]
    return spark.createDataFrame(data, schema)


def test_unify_nodes(spark, sample_nodes):
    # Create two node datasets
    nodes1 = sample_nodes.filter(sample_nodes.id != "MONDO:0005148")
    nodes2 = sample_nodes.filter(sample_nodes.id != "CHEBI:119157")

    # Call the unify_nodes function
    result = nodes.union_and_deduplicate_nodes(["nodes1", "nodes2"], nodes1=nodes1, nodes2=nodes2)

    # Check the result
    assert isinstance(result, DataFrame)
    assert result.count() == 2  # Should have deduplicated
    assert set(result.columns) == set(KGNodeSchema.to_schema().columns)

    # Check if the properties are combined correctly for the duplicated node
    drug_node = result.filter(result.id == "CHEBI:119157").collect()[0]
    assert set(drug_node.all_categories) == {"biolink:Drug", "biolink:ChemicalSubstance", "biolink:SmallMolecule"}
    assert set(drug_node.publications) == {"PMID:12345678", "PMID:34567890"}
    assert set(drug_node.upstream_data_source) == {"source1", "source3"}


def test_unify_edges(spark, sample_edges):
    # Create two edge datasets
    edges1 = sample_edges.filter(sample_edges.subject != "CHEBI:120688")
    edges2 = sample_edges.filter(sample_edges.subject != "CHEBI:119157")

    # Call the unify_edges function
    result = nodes.union_and_deduplicate_edges(["edges1", "edges2"], edges1=edges1, edges2=edges2)

    # Check the result
    assert isinstance(result, DataFrame)
    assert result.count() == 2  # Should have deduplicated
    assert set(result.columns) == set(KGEdgeSchema.to_schema().columns)

    # Check if the properties are combined correctly for the duplicated edge
    treat_edge = result.filter((result.subject == "CHEBI:119157") & (result.object == "MONDO:0005148")).collect()[0]
    assert treat_edge.knowledge_level == "knowledge_assertion"
    assert treat_edge.primary_knowledge_source == "infores:semmeddb"
    assert set(treat_edge.aggregator_knowledge_source) == {"infores:aggregator1", "infores:aggregator3"}
    assert set(treat_edge.publications) == {"PMID:12345678", "PMID:34567890"}
    assert set(treat_edge.upstream_data_source) == {"source1", "source3"}


# Mock the aiohttp ClientSession.post method
class MockResponse:
    def __init__(self, json, status):
        self._json = json
        self.status = status

    async def json(self):
        return self._json

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


def test_normalize_kg(spark, mocker):
    # Given
    sample_nodes = spark.createDataFrame(
        [
            ("CHEBI:119157", "Drug1", "biolink:Drug"),
            ("MONDO:0005148", "Disease1", "biolink:Disease"),
        ],
        ["id", "name", "category"],
    )

    sample_edges = spark.createDataFrame(
        [
            ("CHEBI:119157", "biolink:treats", "MONDO:0005148"),
        ],
        ["subject", "predicate", "object"],
    )

    mocker.patch(
        "aiohttp.ClientSession.post",
        return_value=MockResponse(
            json={
                "CHEBI:119157": {"id": {"identifier": "CHEBI:normalized_119157"}},
                "MONDO:0005148": None,
            },
            status=200,
        ),
    )

    # Expected output DataFrames
    expected_nodes = spark.createDataFrame(
        [
            ("CHEBI:normalized_119157", "Drug1", "biolink:Drug", "CHEBI:119157", True),
            ("MONDO:0005148", "Disease1", "biolink:Disease", "MONDO:0005148", False),
        ],
        ["id", "name", "category", "original_id", "normalization_success"],
    )

    expected_edges = spark.createDataFrame(
        [
            (
                "CHEBI:normalized_119157",
                "biolink:treats",
                "MONDO:0005148",
                "CHEBI:119157",
                "MONDO:0005148",
                True,
                False,
            ),
        ],
        [
            "subject",
            "predicate",
            "object",
            "original_subject",
            "original_object",
            "subject_normalization_success",
            "object_normalization_success",
        ],
    )

    expected_mapping = spark.createDataFrame(
        [
            ("CHEBI:119157", "CHEBI:normalized_119157", True),
            ("MONDO:0005148", "MONDO:0005148", False),
        ],
        ["id", "normalized_id", "normalization_success"],
    )

    # When
    normalized_nodes, normalized_edges, mapping_df = nodes.normalize_kg(sample_nodes, sample_edges, "http://dummy")

    # Then
    assertDataFrameEqual(normalized_nodes[expected_nodes.columns], expected_nodes)
    assertDataFrameEqual(normalized_edges[expected_edges.columns], expected_edges)
    assertDataFrameEqual(mapping_df[expected_mapping.columns], expected_mapping)
