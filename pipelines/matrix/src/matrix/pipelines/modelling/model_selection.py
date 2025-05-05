import numpy as np
from sklearn.model_selection import BaseCrossValidator


class DrugStratifiedSplit(BaseCrossValidator):
    """A cross-validator that provides train/test indices to split data in train/test sets.

    This cross-validator ensures each drug is represented in both training and test sets.
    """

    def __init__(self, n_splits: int = 1, test_size: float = 0.1, random_state: int = None):
        """Initialize the DrugStratifiedSplit cross-validator.

        Args:
            n_splits: Number of re-shuffling & splitting iterations.
            test_size: Proportion of the dataset to include in the test split.
            random_state: Controls the randomness of the training and testing indices produced.
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


class DrugCVSplit(BaseCrossValidator):
    """A cross-validator that provides train/test indices to split data in train/test sets.

    This cross-validator ensures each drug is represented in either the training or test set,
    but never both. A percentage of drugs are pre-selected for the test set, and all their
    data is placed in the test set.

    """

    def __init__(self, n_splits: int = 1, test_size: float = 0.1, random_state: int = None):
        """Initialize the DrugCVSplit cross-validator.

        Args:
            n_splits: Number of re-shuffling & splitting iterations.
            test_size: Proportion of drugs to include in the test split.
            random_state: Controls the randomness of the training and testing indices produced.
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

        # Get unique drug IDs
        unique_drugs = X["source"].unique()
        n_drugs = len(unique_drugs)
        n_test_drugs = max(1, int(np.round(n_drugs * self.test_size)))

        for _ in range(self.n_splits):
            # Randomly select drugs for the test set
            test_drugs = rng.choice(unique_drugs, size=n_test_drugs, replace=False)

            # Initialize train and test indices
            train_indices = []
            test_indices = []

            # For each drug, either add all its data to train or test
            for drug in unique_drugs:
                drug_data = X[X["source"] == drug]
                indices = drug_data.index.tolist()

                if drug in test_drugs:
                    # All data for test drugs goes to test set
                    test_indices.extend(indices)
                else:
                    # All data for non-test drugs goes to train set
                    train_indices.extend(indices)

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
