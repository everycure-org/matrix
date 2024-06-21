"""Module containing classes for evaluation."""
import pandas as pd
import abc
import json
from typing import Dict, List

from refit.v1.core.inject import inject_object


class Evaluation(abc.ABC):
    """An abstract class representing evaluation methods for labelled test data."""

    def evaluate(self, data: pd.DataFrame):
        """Performs evaluation on a dataset.

        Args:
            data: Labelled drug-disease dataset.
        """
        ...


class SimpleClassification(Evaluation):
    """A class representing binary class prediction evaluation metrics."""

    @inject_object()
    def __init__(self, metrics: List[callable], threshold: float = 0.5):
        """Initializes the SimpleClassification instance.

        Args:
            metrics: List of metrics for binary class predictions
            threshold: Threshold value for binary class prediction. Defaults to 0.5.
        """
        self._metrics = metrics
        self._threshold = threshold

    def evaluate(
        self,
        data: pd.DataFrame,
        target_col_name: str = "y",  # TODO: Put standard name into params? used also in pair_generator
        score_col_name: str = "treat score",
    ) -> Dict:
        """Evaluated simple classification metrics on a dataset.

        Args:
            data: Labelled drug-disease dataset.
            target_col_name: Target label column name.
            score_col_name: Probability score column name.
            prediction_suffix: Suffix to add to the predicted class column, defaults to '_pred
        """
        # Binary class predictions and true labels
        y_pred = data[score_col_name].ge(self._threshold)
        y_true = data[target_col_name]

        # Evaluate and report metrics
        report = {}
        for metric in self._metrics:
            report[f"{metric.__name__}"] = metric(y_true, y_pred)
        return json.loads(json.dumps(report, default=float))


# class MRREvaluation(Evaluation):
#     # Does this function need anything else to operate?
#     def evaluate(self, data: pd.DataFrame):
#         # TODO: Implement here
#         return {"evaluation": "hitk"}


# class HitKEvaluation(Evaluation):
#     # Does this function need anything else to operate?
#     def evaluate(self, data: pd.DataFrame):
#         # TODO: Implement here
#         return {"evaluation": "hitk"}
