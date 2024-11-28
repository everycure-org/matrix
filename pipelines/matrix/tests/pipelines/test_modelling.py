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
    print("\n=== Test Data Distribution Analysis ===")

    # The first fold
    fold0 = result[0]
    train_count_0 = len(fold0[fold0["split"] == "TRAIN"])
    test_count_0 = len(fold0[fold0["split"] == "TEST"])
    print("\nFold 0:")
    print(f"Train samples: {train_count_0}")
    print(f"Test samples: {test_count_0}")

    # The second fold
    fold1 = result[1]
    train_count_1 = len(fold1[fold1["split"] == "TRAIN"])
    test_count_1 = len(fold1[fold1["split"] == "TEST"])
    print("\nFold 1:")
    print(f"Train samples: {train_count_1}")
    print(f"Test samples: {test_count_1}")

    # The full dataset
    full_data = result[2]
    print("\nFull Dataset:")
    print(f"Total samples: {len(full_data)}")

    # Test set analysis
    test_indices_fold0 = set(fold0[fold0["split"] == "TEST"].index)
    test_indices_fold1 = set(fold1[fold1["split"] == "TEST"].index)

    print("\n=== Test Set Analysis ===")
    print(f"Test indices in fold 0: {sorted(test_indices_fold0)}")
    print(f"Test indices in fold 1: {sorted(test_indices_fold1)}")

    intersection = test_indices_fold0.intersection(test_indices_fold1)
    print(f"\nOverlap between test sets: {intersection}")

    all_test_indices = test_indices_fold0.union(test_indices_fold1)
    print(f"Combined test indices: {sorted(all_test_indices)}")
    print(f"Total unique test samples: {len(all_test_indices)}")

    # Run the original assertions
    assert len(result) == 3
    assert len(fold0[fold0["split"] == "TRAIN"]) == 10
    assert len(fold0[fold0["split"] == "TEST"]) == 15
    assert all(fold0["iteration"] == 0)

    assert len(fold1[fold1["split"] == "TRAIN"]) == 15
    assert len(fold1[fold1["split"] == "TEST"]) == 10
    assert all(fold1["iteration"] == 1)

    assert len(full_data) == 25
    assert all(full_data["split"] == "TRAIN")
    assert all(full_data["iteration"] == 2)

    # Test set independence assertions
    assert len(test_indices_fold0.intersection(test_indices_fold1)) == 0
    assert len(all_test_indices) == len(sample_data)
    assert all_test_indices == set(range(len(sample_data)))
