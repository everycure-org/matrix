import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from matrix.pipelines.evaluation.nodes import make_test_predictions
from matrix.pipelines.modelling.transformers import FlatArrayTransformer


@pytest.fixture
def mock_graph():
    graph = Mock()
    graph._embeddings = {
        # 2 sources, 8 targets
        "drug_1": [0.1, 0.2, 0.3],
        "drug_2": [0.4, 0.5, 0.6],
        "disease_1": [0.7, 0.8, 0.9],
        "disease_2": [1.0, 1.1, 1.2],
        "disease_3": [1.3, 1.4, 1.5],
        "disease_4": [1.6, 1.7, 1.8],
        "disease_5": [1.9, 2.0, 2.1],
        "disease_6": [2.2, 2.3, 2.4],
        "disease_7": [2.5, 2.6, 2.7],
        "disease_8": [2.8, 2.9, 3.0],
    }
    return graph


@pytest.fixture
def mock_data():
    return pd.DataFrame(
        [
            {"source": "drug_1", "target": "disease_1"},
            {"source": "drug_2", "target": "disease_1"},
            {"source": "drug_1", "target": "disease_2"},
            {"source": "drug_2", "target": "disease_2"},
            {"source": "drug_1", "target": "disease_3"},
            {"source": "drug_2", "target": "disease_3"},
            {"source": "drug_1", "target": "disease_4"},
            {"source": "drug_2", "target": "disease_4"},
            {"source": "drug_1", "target": "disease_5"},
            {"source": "drug_2", "target": "disease_5"},
            {"source": "drug_1", "target": "disease_6"},
            {"source": "drug_2", "target": "disease_6"},
            {"source": "drug_1", "target": "disease_7"},
            {"source": "drug_2", "target": "disease_7"},
            {"source": "drug_1", "target": "disease_8"},
            {"source": "drug_2", "target": "disease_8"},
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
def mock_model():
    model = Mock()
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.1, 0.9]])
    return model


def test_make_test_predictions(
    mock_graph,
    mock_data,
    mock_transformers,
    mock_model,
):
    # Given data, embeddings and a model
    # When we make batched predictions
    result = make_test_predictions(
        graph=mock_graph,
        data=mock_data,
        transformers=mock_transformers,
        model=mock_model,
        features=["source_+", "target_+"],
        score_col_name="score",
        batch_by="target",
    )

    # Then the scores are added for all datapoints in a new column
    # and the model was called the correct number of times
    assert "score" in result.columns
    assert len(result) == 16
    assert mock_model.predict_proba.call_count == 8  # Called 8x due to batching
