"""Module containing classes for evaluation."""
import pandas as pd
import abc
import json
import bisect
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


class SpecificRanking(Evaluation):
    """A class representing ranking metrics for specific axes of the matrix.

    In particular, the class encompasses drug or diseases specific Hit@k and Mean Reciprocal Rank (MRR) metrics.

    TODO: Comment about how the metrics are computed.
    """

    @inject_object()
    def __init__(self, rank_func_lst: List[callable], specific_col: str) -> None:
        """Initializes the SpecificRanking instance.

        Args:
            rank_func_lst: List of functions giving ranking function and ranking function name.
            specific_col: Column to rank over.
                Set to "source" for drug-specific ranking.
                Set to "target" for disease-specific ranking.
        """
        self._rank_func_lst = rank_func_lst
        self._specific_col = specific_col

    def evaluate(
        self,
        data: pd.DataFrame,
        score_col_name: str = "treat score",
    ) -> Dict:
        """Evaluates metrics on a dataset.

        Args:
            data: Labelled drug-disease dataset with probability scores.
            score_col_name: Probability score column name.
        """
        # Get items to loop over
        items_lst = list(data[self._specific_col].unique())

        # Compute average rank of known positives for each item
        ranks_lst = []
        for item in items_lst:
            pairs_for_item = data[data[self._specific_col] == item]
            is_pos = pairs_for_item["y"].eq(1)
            pos_preds = pairs_for_item[is_pos][score_col_name]
            neg_preds = pairs_for_item[~is_pos][score_col_name]
            neg_preds = neg_preds.sort()  # TODO: Potential bottleneck here
            for prob in pos_preds:
                rank = len(neg_preds) - bisect.bisect_left(neg_preds, prob) + 1
            ranks_lst.append(rank)

        # Compute average of rank functions and report metrics
        report = {}
        for rank_func_generator in self._rank_func_lst:
            rank_func, rank_func_name = rank_func_generator()
            ranks_arr = np.array(ranks_lst)
            transformed_rank_lst = rank_func(ranks_arr)
            report[f"{rank_func_name}"] = np.mean(transformed_rank_lst)
        return json.loads(json.dumps(report, default=float))

    @classmethod
    def hitk(k: int):
        """Returns vectorised ranking function for Hit@k.

        The Hit@k metric is the expected value of this function for the ground truth positive ranks.

        Args:
            k: Value for k.

        Returns:
            Tuple containing function and name of metric.
        """
        rank_func = lambda rank: np.where(rank <= k, 1, 0)
        rank_func_name = "Hit@" + str(k)
        return rank_func, rank_func_name

    @classmethod
    def mrr():
        """Returns ranking function for MRR.

        The MRR metric is the expected value of this function for the ground truth positive ranks.

        Returns:
            Tuple containing function and name of metric.
        """
        rank_func = lambda rank: 1 / rank
        rank_func_name = "MRR"
        return rank_func, rank_func_name


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
