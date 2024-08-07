"""Module containing classes for evaluation."""
import pandas as pd
import numpy as np
import abc
import json
import bisect
from typing import Dict, List


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

    def __init__(self, metrics: dict, score_col_name: str, threshold: float = 0.5):
        """Initializes the DiscreteMetrics instance.

        Args:
            metrics: List of metrics for binary class predictions
            score_col_name: Probability score column name.
            threshold: Threshold value for binary class prediction. Defaults to 0.5.
        """
        self._metrics = metrics
        self._threshold = threshold
        self._score_col_name = score_col_name

    def evaluate(
        self,
        data: pd.DataFrame,
    ) -> Dict:
        """Evaluates metrics on a dataset.

        Args:
            data: Labelled drug-disease dataset with probability scores.
            target_col_name: Target label column name.
        """
        # Binary class predictions and true labels
        y_pred = data[self._score_col_name].ge(self._threshold)
        y_true = data["y"]

        # Evaluate and report metrics
        report = {}
        for metric in self._metrics:
            report[f"{metric.__name__}"] = metric(y_true, y_pred)
        return json.loads(json.dumps(report, default=float))


class ContinuousMetrics(Evaluation):
    """A class representing metrics evaluating continuous binary class probability scores."""

    def __init__(self, metrics: List[callable], score_col_name: str):
        """Initializes the ContinuousMetrics instance.

        Args:
            metrics: List of metrics for binary class predictions
            score_col_name: Probability score column name.
        """
        self._metrics = metrics
        self._score_col_name = score_col_name

    def evaluate(
        self,
        data: pd.DataFrame,
    ) -> Dict:
        """Evaluates metrics on a dataset.

        Args:
            data: Labelled drug-disease dataset with probability scores.

        """
        # Binary class predictions and true labels
        y_score = data[self._score_col_name]
        y_true = data["y"]

        # Evaluate and report metrics
        report = {}
        for metric in self._metrics:
            report[f"{metric.__name__}"] = metric(y_true, y_score)
        return json.loads(json.dumps(report, default=float))


class SpecificRanking(Evaluation):
    """A class representing ranking metrics for specific axes of the matrix.

    In particular, the class encompasses drug or diseases specific Hit@k and Mean Reciprocal Rank (MRR) metrics.

    Note that, for each specific drug or disease, we compute the rank of each known positives only against negatives,
    not including the other known positives.

    TODO: unit test
    """

    def __init__(
        self, rank_func_lst: List[callable], specific_col: str, score_col_name: str
    ) -> None:
        """Initializes the SpecificRanking instance.

        Args:
            rank_func_lst: List of functions giving ranking function and ranking function name.
            specific_col: Column to rank over.
                Set to "source" for drug-specific ranking.
                Set to "target" for disease-specific ranking.
            score_col_name: Probability score column name.
        """
        self._rank_func_lst = rank_func_lst
        self._specific_col = specific_col
        self._score_col_name = score_col_name

    def evaluate(
        self,
        data: pd.DataFrame,
    ) -> Dict:
        """Evaluates metrics on a dataset.

        Args:
            data: Labelled drug-disease dataset with probability scores.
        """
        # Get items to loop over
        items_lst = list(data[self._specific_col].unique())

        # Compute ranks of known positives for each item
        ranks_lst = []
        for item in items_lst:
            pairs_for_item = data[data[self._specific_col] == item]
            is_pos = pairs_for_item["y"].eq(1)
            pos_preds = list(pairs_for_item[is_pos][self._score_col_name])
            neg_preds = list(pairs_for_item[~is_pos][self._score_col_name])
            neg_preds.sort()

            for prob in pos_preds:
                rank = len(neg_preds) - bisect.bisect_left(neg_preds, prob) + 1
                ranks_lst.append(rank)

        # Compute average of rank functions and report metrics
        report = {}
        for rank_func_generator in self._rank_func_lst:
            rank_func = rank_func_generator.generate()
            ranks_arr = np.array(ranks_lst)
            transformed_rank_lst = rank_func(ranks_arr)
            report[f"{rank_func_generator.name()}"] = np.mean(transformed_rank_lst)
        return json.loads(json.dumps(report, default=float))


class RankingFunction(abc.ABC):
    """Class generating a vectorised function used in the computation of ranking-based evaluation metrics."""

    def generate(self):
        """Returns function."""
        ...

    def name(self):
        """Returns name of the function."""
        ...


class MRR(RankingFunction):
    """Class generating a vectorised function for the computation of MRR."""

    @staticmethod
    def generate():
        """Returns function."""
        return lambda rank: 1 / rank

    @staticmethod
    def name():
        """Returns name of the function."""
        return "mrr"


class HitK(RankingFunction):
    """Class generating a vectorised function for the computation of Hit@k."""

    def __init__(self, k) -> None:
        """Initialise instance of Hitk object.

        Args:
            k: Value for k.
        """
        self.k = k

    def generate(self):
        """Returns function."""
        return lambda rank: np.where(rank <= self.k, 1, 0)

    def name(self):
        """Returns name of the function."""
        return "hit-" + str(self.k)
