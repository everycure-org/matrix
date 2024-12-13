import pandas as pd
import pytest
import pandera
from matrix.pipelines.preprocessing.nodes import create_int_edges, create_prm_nodes


@pytest.fixture
def sample_int_nodes():
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "name": ["Drug A", "Disease B", "Drug C", "Disease D"],
            "normalized_curie": ["MESH:123", "MONDO:456", None, "MONDO:789"],
        }
    )


@pytest.fixture
def sample_edges():
    return pd.DataFrame({"Source": [1, 2, 3, 4], "Target": [2, 3, 4, 1], "other_col": ["a", "b", "c", "d"]})


def test_create_int_edges_basic(sample_int_nodes: pd.DataFrame, sample_edges: pd.DataFrame) -> None:
    result = create_int_edges(sample_int_nodes, sample_edges)

    # Check output columns
    assert set(["SourceId", "TargetId", "Included"]).issubset(result.columns)

    # Check that edges are properly mapped
    assert result.iloc[0]["SourceId"] == "MESH:123"
    assert result.iloc[0]["TargetId"] == "MONDO:456"
    assert result.iloc[0]["Included"] == True  # noqa: E712


def test_create_int_edges_with_missing_nodes(sample_int_nodes: pd.DataFrame, sample_edges: pd.DataFrame) -> None:
    # Edge with node that has no normalized_curie
    result = create_int_edges(sample_int_nodes, sample_edges)

    # Row involving node 3 (which has no normalized_curie) should have Included=False
    row_with_missing = result[result["Source"] == 3].iloc[0]
    assert pd.isna(row_with_missing["SourceId"])
    assert row_with_missing["Included"] == False  # noqa: E712


def test_create_int_edges_empty_input() -> None:
    empty_nodes = pd.DataFrame({"ID": [], "name": [], "normalized_curie": []})
    empty_edges = pd.DataFrame({"Source": [], "Target": []})

    result = create_int_edges(empty_nodes, empty_edges)
    assert len(result) == 0
    assert set(["SourceId", "TargetId", "Included"]).issubset(result.columns)


def test_create_int_edges_preserves_other_columns(sample_int_nodes: pd.DataFrame, sample_edges: pd.DataFrame) -> None:
    result = create_int_edges(sample_int_nodes, sample_edges)

    # Check that non-mapping columns are preserved
    assert "other_col" in result.columns
    assert list(result["other_col"]) == ["a", "b", "c", "d"]


def test_create_prm_nodes_valid_input():
    """Test create_prm_nodes with valid input data."""
    # Arrange
    input_df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "normalized_curie": ["MESH:123", "MONDO:456", "HP:789"],
            "entity label": ["Gene", "Disease", "PhenotypicFeature"],
            "name": ["Gene1", "Disease1", "Phenotype1"],
            "description": ["desc1", "desc2", "desc3"],
        }
    )

    # Act
    result = create_prm_nodes(input_df)

    # Assert
    expected_df = pd.DataFrame(
        {
            "id": ["MESH:123", "MONDO:456", "HP:789"],
            "entity label": ["Gene", "Disease", "PhenotypicFeature"],
            "category": ["biolink:Gene", "biolink:Disease", "biolink:PhenotypicFeature"],
            "name": ["Gene1", "Disease1", "Phenotype1"],
            "description": ["desc1", "desc2", "desc3"],
        }
    )
    pd.testing.assert_frame_equal(result, expected_df, check_like=True)


def test_create_prm_nodes_handles_null_normalized_curie():
    """Test create_prm_nodes filters out rows with null normalized_curie."""
    # Arrange
    input_df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "normalized_curie": ["MESH:123", None, "HP:789"],
            "entity label": ["Gene", "Disease", "PhenotypicFeature"],
            "name": ["Gene1", "Disease1", "Phenotype1"],
            "description": ["desc1", "desc2", "desc3"],
        }
    )

    # Act
    result = create_prm_nodes(input_df)

    # Assert
    assert len(result) == 2
    assert None not in result["id"].values


def test_create_prm_nodes_handles_duplicates():
    """Test create_prm_nodes removes duplicate IDs."""
    # Arrange
    input_df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "normalized_curie": ["MESH:123", "MESH:123", "HP:789"],
            "entity label": ["Gene", "Gene", "PhenotypicFeature"],
            "name": ["Gene1", "Gene1", "Phenotype1"],
            "description": ["desc1", "desc1", "desc3"],
        }
    )

    # Act
    result = create_prm_nodes(input_df)

    # Assert
    assert len(result) == 2
    assert result["id"].nunique() == len(result)


def test_create_prm_nodes_schema_validation():
    """Test create_prm_nodes schema validation."""
    # Arrange
    input_df = pd.DataFrame(
        {
            "ID": [1],
            "normalized_curie": ["MESH:123"],
            "entity label": ["Gene"],
            # Missing required columns 'name' and 'description'
        }
    )

    # Act/Assert
    with pytest.raises(pandera.errors.SchemaError):
        create_prm_nodes(input_df)


def test_create_prm_nodes_empty_input():
    """Test create_prm_nodes with empty input."""
    # Arrange
    input_df = pd.DataFrame({"ID": [], "normalized_curie": [], "entity label": [], "name": [], "description": []})

    # Act
    result = create_prm_nodes(input_df)

    # Assert
    assert len(result) == 0
    assert list(result.columns) == ["category", "id", "name", "description"]
