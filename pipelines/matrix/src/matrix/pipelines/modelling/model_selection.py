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
