import numpy as np
from sklearn.model_selection import BaseCrossValidator


class DrugStratifiedSplit(BaseCrossValidator):
    """A cross-validator that provides train/test indices to split data in train/test sets.

    This cross-validator ensures each drug is represented in both training and test sets.
    """

    def __init__(
        self, n_splits=1, test_size=0.1, random_state=None, disease_grouping_type=None, holdout_disease_types=None
    ):
        """Initialize the DrugStratifiedSplit cross-validator.

        Args:
            n_splits (int): Number of re-shuffling & splitting iterations.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Controls the randomness of the training and testing indices produced.
            disease_grouping_type (str): The type of disease grouping to use.
            holdout_disease_types (list): The list of disease types to hold out.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.disease_grouping_type = (
            disease_grouping_type  # for consistency with DiseaseAreaSplit, this is not used in DrugStratifiedSplit
        )
        self.holdout_disease_types = (
            holdout_disease_types  # for consistency with DiseaseAreaSplit, this is not used in DrugStratifiedSplit
        )

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


class DiseaseAreaSplit(BaseCrossValidator):
    """A disease area cross-validator that provides train/test indices to split data in train/test sets based on disease area."""

    def __init__(
        self, n_splits=1, test_size=0.1, random_state=None, disease_grouping_type=None, holdout_disease_types=None
    ):
        """Initialize the DiseaseAreaSplit cross-validator.

        Args:
            n_splits (int): Number of re-shuffling & splitting iterations.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Controls the randomness of the training and testing indices produced.
            disease_grouping_type (str): The type of disease grouping to use.
            holdout_disease_types (list): The list of disease types to hold out.
        """
        self.n_splits = n_splits
        self.test_size = test_size  # for consistency with DrugStratifiedSplit, this is not used in DiseaseAreaSplit
        self.random_state = (
            random_state  # for consistency with DrugStratifiedSplit, this is not used in DiseaseAreaSplit
        )
        self.disease_grouping_type = disease_grouping_type
        self.holdout_disease_types = holdout_disease_types

    def split(self, X, disease_list=None):
        """Generate indices to split data into training and test set.

        Args:
            X (pandas.DataFrame): The data to be split.
            disease_list (pandas.DataFrame): The disease list to be used for splitting.

        Yields:
            tuple: (train_indices, test_indices)
        """
        # get neccessary columns from disease list
        disease_grouping_type = self.disease_grouping_type
        holdout_disease_types = self.holdout_disease_types

        if disease_grouping_type in disease_list.columns:
            disease_list_copy = disease_list[["category_class", disease_grouping_type]].copy()
            # merge disease list with data
            X_copy = X.copy()
            X_copy = X_copy.merge(disease_list_copy, left_on="target", right_on="category_class", how="left")
            X_copy = X_copy[~X_copy.category_class.isna()]
        else:
            raise ValueError(f"Disease grouping type {disease_grouping_type} not found in disease_list")

        # get indices of rows where disease type is in holdout_disease_types
        for i in range(self.n_splits):
            selected_disease_types = holdout_disease_types[i]

            # Handle NaN values by filling them with an empty string and then doing the contains check
            mask = X_copy[disease_grouping_type].fillna("").str.contains(selected_disease_types, na=False)
            holdout_indices = X_copy[mask].index.tolist()
            train_indices = X_copy[~mask].index.tolist()

            yield train_indices, holdout_indices

    def get_n_splits(self, X=None, disease_list=None):
        """Returns the number of splitting iterations in the cross-validator.

        Args:
            X: Ignored, present for API consistency with scikit-learn.
            disease_list: Ignored, present for API consistency with scikit-learn.

        Returns:
            int: Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits
