"""This module contains custom cross-validation classes for drug-related data."""

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class DrugStratifiedSplit(BaseCrossValidator):
    """A cross-validator that provides train/test indices to split data in train/test sets.

    This cross-validator ensures each drug is represented in both training and test sets.
    """

    def __init__(self, n_splits=1, test_size=0.1, random_state=None):
        """Initialize the DrugStratifiedSplit cross-validator.

        Args:
            n_splits (int): Number of re-shuffling & splitting iterations.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Controls the randomness of the training and testing indices produced.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Args:
            X (pandas.DataFrame): The data to be split.
            y: Ignored, present for API consistency with scikit-learn.
            groups: Ignored, present for API consistency with scikit-learn.

        Yields:
            tuple: (train_indices, test_indices)
        """
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_splits):
            train_indices, test_indices = [], []

            for _, group in X.groupby("source"):
                indices = group.index.tolist()
                rng.shuffle(indices)
                n = len(indices)
                n_test = max(1, int(np.round(n * self.test_size)))
                n_train = n - n_test

                train_indices.extend(indices[:n_train])
                test_indices.extend(indices[n_train:])

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Args:
            X: Ignored, present for API consistency with scikit-learn.
            y: Ignored, present for API consistency with scikit-learn.
            groups: Ignored, present for API consistency with scikit-learn.

        Returns:
            int: Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class GroupAwareSplit(BaseCrossValidator):
    """A cross-validator that provides train/test indices to split data in train/test sets.

    This cross-validator ensures that the values of a chosen feature do not overlap between the test and train splits.
    """

    def __init__(
        self, group_by_column: str = "source_id", n_splits: int = 1, test_size: float = 0.1, random_state: float = None
    ) -> None:
        """Initialize the GroupAwareSplit cross-validator.

        Args:
            group_by_column: The column name to use for grouping the data.
            n_splits: Number of re-shuffling & splitting iterations.
            test_size: Proportion of the dataset to include in the test split.
            random_state: Controls the randomness of the training and testing indices produced.
        """
        self.group_by_column = group_by_column
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Args:
            X (pandas.DataFrame): The data to be split.
            y: Ignored, present for API consistency with scikit-learn.
            groups: Ignored, present for API consistency with scikit-learn.

        Yields:
            tuple: (train_indices, test_indices)
        """
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_splits):
            unique_groups = X[self.group_by_column].unique()
            rng.shuffle(unique_groups)

            n_groups = len(unique_groups)
            n_test_groups = max(1, int(np.round(n_groups * self.test_size)))

            test_groups = unique_groups[:n_test_groups]
            train_groups = unique_groups[n_test_groups:]

            train_indices = X[X[self.group_by_column].isin(train_groups)].index.tolist()
            test_indices = X[X[self.group_by_column].isin(test_groups)].index.tolist()

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Args:
            X: Ignored, present for API consistency with scikit-learn.
            y: Ignored, present for API consistency with scikit-learn.
            groups: Ignored, present for API consistency with scikit-learn.

        Returns:
            int: Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits
