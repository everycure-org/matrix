import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from matrix.pipelines.matrix_generation.nodes import (
    generate_pairs,
    make_predictions_and_sort,
    generate_report,
)
from matrix.pipelines.modelling.transformers import FlatArrayTransformer
from matrix.datasets.graph import KnowledgeGraph
from matrix.pipelines.modelling.model import ModelWrapper


@pytest.fixture
def sample_data():
    """Fixture that provides sample data for testing matrix generation functions."""
    drugs = pd.DataFrame(
        {
            "curie": ["drug_1", "drug_2"],
            "name": ["Drug 1", "Drug 2"],
            "description": ["Description 1", "Description 2"],
        }
    )

    diseases = pd.DataFrame(
        {
            "curie": ["disease_1", "disease_2"],
            "name": ["Disease 1", "Disease 2"],
            "description": ["Description A", "Description B"],
        }
    )

    known_pairs = pd.DataFrame(
        {
            "source": ["drug_1", "drug_2", "drug_3"],
            "target": ["disease_1", "disease_2", "disease_3"],
            "split": ["TRAIN", "TEST", "TRAIN"],
            "y": [1, 0, 1],
        }
    )

    nodes = pd.DataFrame(
        {
            "id": ["drug_1", "drug_2", "disease_1", "disease_2"],
            "is_drug": [True, True, False, False],
            "is_disease": [False, False, True, True],
            "topological_embedding": [np.ones(3) * n for n in range(4)],
        }
    )
    graph = KnowledgeGraph(nodes)

    return drugs, diseases, known_pairs, graph


def test_generate_pairs(sample_data):
    """Test the generate_pairs function."""
    # Given drug list, disease list and known pairs
    drugs, diseases, known_pairs, _ = sample_data

    # When generating the matrix dataset
    result = generate_pairs(drugs, diseases, known_pairs)

    # Then the output is of the correct format, shape and doesn't contain training pairs
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"source", "target"}
    assert len(result) == 3  # 2 drugs * 2 diseases - 1 training pair
    assert not result.apply(
        lambda row: (row["source"], row["target"]) in [("drug_1", "disease_1")], axis=1
    ).any()


@pytest.fixture
def mock_transformers():
    return {
        "flat_source_array": {
            "transformer": FlatArrayTransformer(prefix="source_"),
            "features": ["source_embedding"],
        },
        "flat_target_array": {
            "transformer": FlatArrayTransformer(prefix="target_"),
            "features": ["target_embedding"],
        },
    }


@pytest.fixture
def mock_data():
    return pd.DataFrame(
        [
            {"source": "drug_1", "target": "disease_1"},
            {"source": "drug_2", "target": "disease_1"},
            {"source": "drug_1", "target": "disease_2"},
            {"source": "drug_2", "target": "disease_2"},
        ]
    )


@pytest.fixture
def mock_transformers():
    return {
        "flat_source_array": {
            "transformer": FlatArrayTransformer(prefix="source_"),
            "features": ["source_embedding"],
        },
        "flat_target_array": {
            "transformer": FlatArrayTransformer(prefix="target_"),
            "features": ["target_embedding"],
        },
    }


@pytest.fixture
def mock_model():  # Note: gives correct shaped output only for batches of size 2
    model = Mock()
    model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1], [0.1, 0.7, 0.2]])
    return model


def test_make_predictions_and_sort(
    sample_data,
    mock_data,
    mock_transformers,
    mock_model,
):
    # Given a drug list, disease list and objects necessary for inference
    # When running inference and sorting
    _, _, _, graph = sample_data
    result = make_predictions_and_sort(
        graph=graph,
        data=mock_data,
        transformers=mock_transformers,
        model=mock_model,
        features=["source_+", "target_+"],
        score_col_name="score",
        batch_by="target",
    )

    # Then the output is of the correct format and correctly sorted
    assert "score" in result.columns
    assert isinstance(result, pd.DataFrame)
    assert result["score"].is_monotonic_decreasing


def test_generate_report(sample_data):
    """Test the generate_report function."""
    # Given an input matrix, drug list and disease list
    drugs, diseases, known_pairs, _ = sample_data
    data = pd.DataFrame(
        {
            "source": ["drug_1", "drug_2", "drug_1"],
            "target": ["disease_1", "disease_2", "disease_2"],
            "probability": [0.8, 0.6, 0.4],
        }
    )
    n_reporting = 2

    # When generating the report
    result = generate_report(
        data, n_reporting, drugs, diseases, known_pairs, "probability"
    )

    # Then the report is of the correct structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == n_reporting
    assert set(result.columns) == {
        "drug_id",
        "drug_name",
        "disease_id",
        "disease_name",
        "probability",
        "is_known_positive",
        "is_known_negative",
    }
    assert result["drug_name"].tolist() == ["Drug 1", "Drug 2"]
    assert result["disease_name"].tolist() == ["Disease 1", "Disease 2"]
