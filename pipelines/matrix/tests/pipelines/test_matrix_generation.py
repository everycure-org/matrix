import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from matrix.pipelines.matrix_generation.nodes import (
    generate_pairs,
    make_batch_predictions,
    make_predictions_and_sort,
    generate_report,
)
from matrix.pipelines.modelling.transformers import FlatArrayTransformer
from matrix.datasets.graph import KnowledgeGraph


@pytest.fixture
def sample_drugs():
    """Fixture that provides sample drugs data for testing."""
    return pd.DataFrame(
        {
            "curie": ["drug_1", "drug_2"],
            "name": ["Drug 1", "Drug 2"],
            "description": ["Description 1", "Description 2"],
        }
    )


@pytest.fixture
def sample_diseases():
    """Fixture that provides sample diseases data for testing."""
    return pd.DataFrame(
        {
            "curie": ["disease_1", "disease_2"],
            "name": ["Disease 1", "Disease 2"],
            "description": ["Description A", "Description B"],
        }
    )


@pytest.fixture
def sample_known_pairs():
    """Fixture that provides sample known pairs data for testing."""
    return pd.DataFrame(
        {
            "source": ["drug_1", "drug_2", "drug_3"],
            "target": ["disease_1", "disease_2", "disease_3"],
            "split": ["TRAIN", "TEST", "TRAIN"],
            "y": [1, 0, 1],
        }
    )


@pytest.fixture
def sample_clinical_trials():
    """Fixture that provides sample clinical trials data for testing."""
    return pd.DataFrame(
        {
            "source": ["drug_1"],
            "target": ["disease_2"],
            "significantly_better": [1],
            "non_significantly_better": [0],
            "non_significantly_worse": [0],
            "significantly_worse": [0],
        }
    )


@pytest.fixture
def sample_graph():
    """Fixture that provides a sample KnowledgeGraph for testing."""
    nodes = pd.DataFrame(
        {
            "id": ["drug_1", "drug_2", "disease_1", "disease_2"],
            "is_drug": [True, True, False, False],
            "is_disease": [False, False, True, True],
            "topological_embedding": [np.ones(3) * n for n in range(4)],
        }
    )
    return KnowledgeGraph(nodes)


@pytest.fixture
def sample_matrix_data():
    return pd.DataFrame(
        [
            {"source": "drug_1", "target": "disease_1"},
            {"source": "drug_2", "target": "disease_1"},
            {"source": "drug_1", "target": "disease_2"},
            {"source": "drug_2", "target": "disease_2"},
        ]
    )


@pytest.fixture
def transformers():
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


@pytest.fixture
def mock_model_2():
    model = Mock()
    model.predict_proba = lambda x: np.array([[1, 0, 0]] * len(x))
    return model


def test_generate_pairs(sample_drugs, sample_diseases, sample_known_pairs, sample_clinical_trials):
    """Test the generate_pairs function."""
    # Given drug list, disease list and ground truth pairs
    # When generating the matrix dataset
    result = generate_pairs(sample_drugs, sample_diseases, sample_known_pairs, sample_clinical_trials)

    # Then the output is of the correct format and shape
    assert isinstance(result, pd.DataFrame)
    assert {"source", "target"}.issubset(set(result.columns))
    assert len(result) == 3  # 2 drugs * 2 diseases - 1 training pair
    # Doesn't contain training pairs
    assert not result.apply(lambda row: (row["source"], row["target"]) in [("drug_1", "disease_1")], axis=1).any()
    # Boolean flag columns are present
    check_col = lambda col_name: (col_name in result.columns) and result[col_name].dtype == bool
    assert all(
        check_col(col)
        for col in [
            "is_known_positive",
            "is_known_negative",
            "trial_sig_better",
            "trial_non_sig_better",
            "trial_sig_worse",
            "trial_non_sig_worse",
        ]
    )
    # Flag columns set correctly
    assert all(
        [
            sum(result[col_name]) == 1
            for col_name in (
                "is_known_negative",
                "trial_sig_better",
            )
        ]
    )
    assert all(
        [
            sum(result[col_name]) == 0
            for col_name in (
                "is_known_positive",
                "trial_non_sig_better",
                "trial_sig_worse",
                "trial_non_sig_worse",
            )
        ]
    )


def test_make_batch_predictions(
    sample_graph,
    sample_matrix_data,
    transformers,
    mock_model,
):
    # Given data, embeddings and a model
    # When we make batched predictions
    result = make_batch_predictions(
        graph=sample_graph,
        data=sample_matrix_data,
        transformers=transformers,
        model=mock_model,
        features=["source_+", "target_+"],
        score_col_name="score",
        batch_by="target",
    )

    # Then the scores are added for all datapoints in a new column
    # and the model was called the correct number of times
    assert "score" in result.columns
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert mock_model.predict_proba.call_count == 2  # Called 2x due to batching


def test_make_predictions_and_sort(
    sample_graph,
    sample_matrix_data,
    transformers,
    mock_model,
):
    # Given a drug list, disease list and objects necessary for inference
    # When running inference and sorting
    result = make_predictions_and_sort(
        graph=sample_graph,
        data=sample_matrix_data,
        transformers=transformers,
        model=mock_model,
        features=["source_+", "target_+"],
        score_col_name="score",
        batch_by="target",
    )

    # Then the output is of the correct format and correctly sorted
    assert "score" in result.columns
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert result["score"].is_monotonic_decreasing


def test_generate_report(
    sample_graph,
    transformers,
    mock_model_2,
    sample_drugs,
    sample_diseases,
    sample_known_pairs,
    sample_clinical_trials,
):
    """Test the generate_report function."""
    # Given an input matrix, drug list and disease list
    matrix = generate_pairs(sample_drugs, sample_diseases, sample_known_pairs, sample_clinical_trials)
    matrix_with_scores = make_predictions_and_sort(
        graph=sample_graph,
        data=matrix,
        transformers=transformers,
        model=mock_model_2,
        features=["source_+", "target_+"],
        score_col_name="score",
        batch_by="target",
    )
    n_reporting = 2

    # When generating the report
    result = generate_report(matrix_with_scores, n_reporting, sample_drugs, sample_diseases, "score")

    # Then the report is of the correct structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == n_reporting
    assert set(result.columns) == {
        "drug_id",
        "drug_name",
        "disease_id",
        "disease_name",
        "score",
        "is_known_positive",
        "is_known_negative",
    }
    assert set(result["drug_name"]).issubset({"Drug 1", "Drug 2"})
    assert set(result["disease_name"]).issubset({"Disease 1", "Disease 2"})
