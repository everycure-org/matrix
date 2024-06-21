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


class DiscreteMetrics(Evaluation):
    """A class representing metrics evaluating discrete binary class prediction."""

    @inject_object()
    def __init__(self, metrics: List[callable], threshold: float = 0.5):
        """Initializes the DiscreteMetrics instance.

        Args:
            metrics: List of metrics for binary class predictions
            threshold: Threshold value for binary class prediction. Defaults to 0.5.
        """
        self._metrics = metrics
        self._threshold = threshold

    def evaluate(
        self,
        data: pd.DataFrame,
        score_col_name: str = "treat score",
    ) -> Dict:
        """Evaluates metrics on a dataset.

        Args:
            data: Labelled drug-disease dataset with probability scores.
            target_col_name: Target label column name.
            score_col_name: Probability score column name.
        """
        # Binary class predictions and true labels
        y_pred = data[score_col_name].ge(self._threshold)
        y_true = data["y"]

        # Evaluate and report metrics
        report = {}
        for metric in self._metrics:
            report[f"{metric.__name__}"] = metric(y_true, y_pred)
        return json.loads(json.dumps(report, default=float))


class ContinuousMetrics(Evaluation):
    """A class representing metrics evaluating continuous binary class probability scores."""

    @inject_object()
    def __init__(self, metrics: List[callable]):
        """Initializes the ContinuousMetrics instance.

        Args:
            metrics: List of metrics for binary class predictions
        """
        self._metrics = metrics
        self._threshold = threshold

    def evaluate(
        self,
        data: pd.DataFrame,
        score_col_name: str = "treat score",
    ) -> Dict:
        """Evaluates metrics on a dataset.

        Args:
            data: Labelled drug-disease dataset with probability scores.
            target_col_name: Target label column name.
            score_col_name: Probability score column name.
        """
        # Binary class predictions and true labels
        y_score = data[score_col_name]
        y_true = data["y"]

        # Evaluate and report metrics
        report = {}
        for metric in self._metrics:
            report[f"{metric.__name__}"] = metric(y_true, y_score)
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
