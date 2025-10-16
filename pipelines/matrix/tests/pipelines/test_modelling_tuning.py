import numpy as np
from matrix.pipelines.modelling.tuning import GaussianSearch, NopTuner
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from skopt.space import Integer, Real


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


def test_gaussian_search_parallel():
    """Test GaussianSearch with parallel evaluation support."""

    # Create a simple estimator with n_jobs=-1
    class ParallelEstimator(BaseEstimator):
        def __init__(self, param1=1, param2=0.5, n_jobs=1, random_state=42):
            self.param1 = param1
            self.param2 = param2
            self.n_jobs = n_jobs
            self.random_state = random_state

        def fit(self, X, y=None):
            return self

        def predict(self, X):
            # Simple prediction based on param1
            return np.full(len(X), self.param1 % 2)

    # Create synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    # Define search space
    dimensions = [
        Integer(name="param1", low=1, high=5),
        Real(name="param2", low=0.1, high=1.0),
    ]

    # Create tuner with n_jobs=-1 to test parallel evaluation
    estimator = ParallelEstimator(n_jobs=-1, random_state=42)
    tuner = GaussianSearch(
        estimator=estimator,
        dimensions=dimensions,
        scoring=accuracy_score,
        splitter=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        n_calls=5,  # Small number for quick test
    )

    # Fit tuner
    fitted_estimator = tuner.fit(X, y)

    # Verify results
    assert fitted_estimator is not None
    assert hasattr(tuner, "best_params_")
    assert "param1" in tuner.best_params_
    assert "param2" in tuner.best_params_
    assert hasattr(tuner, "convergence_plot")

    # Verify parameters are within bounds
    assert 1 <= tuner.best_params_["param1"] <= 5
    assert 0.1 <= tuner.best_params_["param2"] <= 1.0
