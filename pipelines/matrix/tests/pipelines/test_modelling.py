import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from matrix.pipelines.modelling.nodes import make_splits
from matrix.pipelines.modelling.model import ModelWrapper


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "source": ["drug1", "drug1", "drug2", "drug2", "drug3"] * 5,
            "target": ["disease1", "disease2", "disease3", "disease4", "disease5"] * 5,
            "source_embedding": [np.ones(1)] * 25,
            "target_embedding": [np.ones(1)] * 25,
            "y": [1, 0, 1, 0, 1] * 5,
        }
    )


def test_make_splits(sample_data, mocker):
    # Mock the settings
    mock_settings = {"DYNAMIC_PIPELINES_MAPPING": {"cross_validation": {"n_splits": 2}}}
    mocker.patch("matrix.settings", mock_settings)

    # Create a simple splitter that splits data into two folds
    class MockSplitter:
        def __init__(self):
            self.n_splits = 2  # Add n_splits attribute required by the function

        def split(self, X, y):
            # First fold: first half train, second half test
            fold1 = (list(range(0, 10)), list(range(10, 25)))
            # Second fold: second half train, first half test
            fold2 = (list(range(10, 25)), list(range(0, 10)))
            return [fold1, fold2]

    # Given a splitter with 2 splits
    splitter = MockSplitter()

    # When we make splits
    result = make_splits(sample_data, splitter)

    # Then we expect the result to be a pandas DataFrame with proper splits
    assert isinstance(result, pd.DataFrame)

    # Check that we have the correct number of rows (original data * 3 due to 2 folds + full dataset)
    assert len(result) == len(sample_data) * 3

    # Verify fold structure
    fold_counts = result["fold"].value_counts()
    assert fold_counts[0] == len(sample_data)  # First fold
    assert fold_counts[1] == len(sample_data)  # Second fold
    assert fold_counts[2] == len(sample_data)  # Full dataset fold

    # Verify splits for fold 0
    fold0_data = result[result["fold"] == 0]
    assert len(fold0_data[fold0_data["split"] == "TRAIN"]) == 10
    assert len(fold0_data[fold0_data["split"] == "TEST"]) == 15

    # Verify splits for fold 1
    fold1_data = result[result["fold"] == 1]
    assert len(fold1_data[fold1_data["split"] == "TRAIN"]) == 15
    assert len(fold1_data[fold1_data["split"] == "TEST"]) == 10

    # Verify full dataset fold (fold 2)
    full_data = result[result["fold"] == 2]
    assert len(full_data) == len(sample_data)
    assert all(full_data["split"] == "TRAIN")


def test_model_wrapper():
    class MyEstimator(BaseEstimator):
        def __init__(self, proba):
            self.proba = proba
            super().__init__()

        def predict_proba(self, X):
            return self.proba

    my_estimators = [
        MyEstimator(proba=[1, 2, 3]),
        MyEstimator(proba=[2, 3, 5]),
    ]

    # given an instance of a model wrapper with mean
    model_mean = ModelWrapper(estimators=my_estimators, agg_func=np.mean)
    # when invoking the predict_proba
    proba_mean = model_mean.predict_proba([])
    # then median computed correctly
    assert np.all(proba_mean == [1.5, 2.5, 4.0])
