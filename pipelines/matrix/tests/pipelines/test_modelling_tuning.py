import numpy as np
from sklearn.base import BaseEstimator
from matrix.pipelines.modelling.tuning import NopTuner


class DummyEstimator(BaseEstimator):
    def __init__(self, param1=1, param2="test"):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_nop_tuner_init():
    estimator = DummyEstimator()
    tuner = NopTuner(estimator)
    assert tuner.estimator == estimator


def test_nop_tuner_init_with_kwargs():
    tuner = NopTuner(DummyEstimator(), param1=42, param2="custom")
    assert isinstance(tuner.estimator, DummyEstimator)


def test_nop_tuner_fit():
    estimator = DummyEstimator(param1=42, param2="test")
    tuner = NopTuner(estimator)

    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    fitted_estimator = tuner.fit(X, y)

    assert fitted_estimator == estimator
    assert tuner.best_params_ == {"param1": 42, "param2": "test"}


def test_nop_tuner_fit_with_params():
    estimator = DummyEstimator()
    tuner = NopTuner(estimator)

    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    fitted_estimator = tuner.fit(X, y, extra_param=True)

    assert fitted_estimator == estimator
    assert tuner.best_params_ == {"param1": 1, "param2": "test"}


def test_nop_tuner_repr() -> None:
    estimator = DummyEstimator()
    tuner = NopTuner(estimator)
    assert repr(tuner) == "NopTuner(estimator=DummyEstimator())"
