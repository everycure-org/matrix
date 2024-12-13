import pytest

from matrix.pipelines.embeddings.nodes import ingest_nodes


@pytest.fixture
def sample_input_df(spark):
    data = [
        {
            "id": "1",
            "name": "Test Node",
            "category": "TestCategory",
            "description": "Test Description",
            "upstream_data_source": ["source1", "source2"],
        },
        {
            "id": "2",
            "name": "Test Node 2",
            "category": "TestCategory2",
            "description": "Test Description 2",
            "upstream_data_source": ["source3"],
        },
    ]
    return spark.createDataFrame(data)


def test_ingest_nodes_basic(sample_input_df):
    """Test basic functionality of ingest_nodes."""
    result = ingest_nodes(sample_input_df)

    # Check schema
    assert "label" in result.columns
    assert "property_keys" in result.columns
    assert "property_values" in result.columns
    assert "array_property_keys" in result.columns
    assert "array_property_values" in result.columns

    # Convert to pandas for easier assertions
    result_pd = result.toPandas()

    # Check first row
    assert result_pd.iloc[0]["label"] == "TestCategory"
    assert set(result_pd.iloc[0]["property_keys"]) == {"name", "category", "description"}
    assert result_pd.iloc[0]["property_values"][0] == "Test Node"
    assert result_pd.iloc[0]["array_property_keys"] == ["upstream_data_source"]
    assert result_pd.iloc[0]["array_property_values"][0] == ["source1", "source2"]


def test_ingest_nodes_null_values(spark):
    """Test handling of null values."""
    data = [
        {
            "id": "1",
            "name": None,
            "category": "TestCategory",
            "description": None,
            "upstream_data_source": None,
        }
    ]
    input_df = spark.createDataFrame(data)

    result = ingest_nodes(input_df)
    result_pd = result.toPandas()

    # Check that nulls are handled properly
    assert result_pd.iloc[0]["property_values"][1] == "TestCategory"  # category should exist
    assert result_pd.iloc[0]["property_values"][0] is None  # name should be null


def test_ingest_nodes_empty_df(spark):
    """Test handling of empty dataframe."""
    empty_df = spark.createDataFrame(
        [], "id string, name string, category string, description string, upstream_data_source array<string>"
    )

    result = ingest_nodes(empty_df)
    assert result.count() == 0
