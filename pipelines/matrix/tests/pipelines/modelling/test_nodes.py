import pytest

import pandas as pd
from sklearn.base import BaseEstimator

from matrix.pipelines.modelling.nodes import train_model


@pytest.fixture("dataset")
def dataset_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [[1, "TRAIN"], [2, "TRAIN"], [3, "TEST"], [4, "TEST"]],
        columns=["iteration", "split"],
    )


class MockEstimator(BaseEstimator):
    def fit(X, y):
        return y


def test_train_model(dataset: pd.DataFrame):
    # Given a dummy estimator instance
    estimator = MockEstimator()

    # When invoking the train model node
    result = train_model(
        dataset, estimator, features=["iteration"], target_col_name="split"
    )

    # Then correct slicing of input dataframe performed
    assert result == pd.DataFrame(
        [
            [1, "TRAIN"],
            [2, "TRAIN"],
        ],
    )
