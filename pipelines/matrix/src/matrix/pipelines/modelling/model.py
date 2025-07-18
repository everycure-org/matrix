from typing import Callable, List, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from xgboost import Booster, XGBClassifier, XGBModel


class ModelWrapper:
    """Class to represent models.

    FUTURE: Add `features` and `transformers` to class, such that we can
    clean up the Kedro viz UI.
    """

    def __init__(
        self,
        estimators: List[BaseEstimator],
        agg_func: Callable,
    ) -> None:
        """Create instance of the model wrapper.

        Args:
            estimators: list of estimators.
            agg_func: function to aggregate ensemble results.
        """
        self._estimators = estimators
        self._agg_func = agg_func
        super().__init__()

    def fit(self, X, y):
        """Model fit method.

        Args:
            X: input features
            y: label
        """
        raise NotImplementedError("ModelWrapper is used to house fitted estimators")

    def predict(self, X):
        """Returns the predicted class.

        Args:
            X: input features
        """
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        """Method for probability scores of the ModelWrapper.

        Args:
            X: input features

        Returns:
            Aggregated probabilities scores of the individual models.
        """
        all_preds = np.array([estimator.predict_proba(X) for estimator in self._estimators])
        return np.apply_along_axis(self._agg_func, 0, all_preds)

    @property
    def boosters(self) -> List[Booster]:
        """
        Return the underlying `xgboost.Booster` objects for every
        XGBoost‑based estimator in the ensemble.

        Non‑XGBoost estimators are filtered out.
        """
        boosters: Sequence[Booster] = []
        for est in self._estimators:
            if isinstance(est, XGBModel):
                boosters.append(est.get_booster())
        if not boosters:
            raise AttributeError(
                "None of the wrapped estimators exposes a Booster; "
                "SHAP analysis requires at least one XGBoost model."
            )
        return list(boosters)


class XGBClassifierWeighted(XGBClassifier):
    def fit(self, X, y, **kwargs):
        return super().fit(X, y, **kwargs)
