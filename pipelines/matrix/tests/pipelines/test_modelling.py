import pandas as pd
import pytest
from matrix.pipelines.modelling.nodes import make_splits


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "source": ["drug1", "drug1", "drug2", "drug2", "drug3"] * 5,
            "target": ["disease1", "disease2", "disease3", "disease4", "disease5"] * 5,
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
            self.n_splits = None

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

    # Then we get 3 dataframes (2 splits + 1 full dataset)
    assert len(result) == 3

    # The first fold has more train data than test data
    fold0 = result[0]
    assert len(fold0[fold0["split"] == "TRAIN"]) == 10
    assert len(fold0[fold0["split"] == "TEST"]) == 15
    assert all(fold0["iteration"] == 0)

    # The second fold has more train data than test data
    fold1 = result[1]
    assert len(fold1[fold1["split"] == "TRAIN"]) == 15
    assert len(fold1[fold1["split"] == "TEST"]) == 10
    assert all(fold1["iteration"] == 1)

    # The full dataset is all train data
    full_data = result[2]
    assert len(full_data) == 25
    assert all(full_data["split"] == "TRAIN")
    assert all(full_data["iteration"] == 2)
