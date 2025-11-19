import numpy as np
from sklearn.model_selection import BaseCrossValidator


class DrugStratifiedSplit(BaseCrossValidator):
    """A cross-validator that provides train/test indices to split data in train/test sets.

    This cross-validator ensures each drug is represented in both training and test sets.
    """

    def __init__(self, n_splits=1, test_size=0.1, random_state=None, **kwargs):
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


class DiseaseAreaSplit(BaseCrossValidator):
    """A disease area cross-validator that provides train/test indices to split data in train/test sets based on disease area."""

    def __init__(
        self,
        n_splits=1,
        random_state=None,
        disease_grouping_type=None,
        holdout_disease_types=None,
        **kwargs,
    ):
        """Initialize the DiseaseAreaSplit cross-validator.

        Args:
            n_splits (int): Number of re-shuffling & splitting iterations.
            random_state (int): Controls the randomness of the training and testing indices produced.
            disease_grouping_type (str): The type of disease grouping to use.
            holdout_disease_types (list): The list of disease types to hold out.

        Example1:
            disease_grouping_type = 'harrisons_view'
            holdout_disease_types = ['cancer_or_benign_tumor', 'inflammatory_disease', 'hereditary_disease', 'syndromic_disease', 'metabolic_disease']
        Example2:
            disease_grouping_type = 'mondo_top_grouping'
            holdout_disease_types = ['infectious_disease', 'cardiovascular_disorder', 'nervous_system_disorder', 'respiratory_system_disorder', 'psychiatric_disorder']
        Example3:
            disease_grouping_type = 'mondo_txgnn'
            holdout_disease_types = ['cancer_or_benign_tumor', 'infectious_disease', 'psychiatric_disorder', 'cardiovascular_disorder', 'autoimmune_disease']
        Example4:
            disease_grouping_type = 'txgnn'
            holdout_disease_types = ['mental_health_disorder', 'cancer', 'cardiovascular_disorder', 'inflammatory_disease', 'neurodegenerative_disease']
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.disease_grouping_type = disease_grouping_type
        self.holdout_disease_types = holdout_disease_types

        # check if length of holdout_disease_types is greater than or equal to n_splits
        if len(self.holdout_disease_types) < self.n_splits:
            raise ValueError(
                f"Length of holdout_disease_types is less than n_splits: {len(self.holdout_disease_types)} < {self.n_splits}"
            )

    def split(self, X, disease_list):
        """Generate indices to split data into training and test set.

        Args:
            X (pandas.DataFrame): The data to be split.
            disease_list (pandas.DataFrame): The disease list to be used for splitting.

        Yields:
            tuple: (train_indices, test_indices)
        """

        if self.disease_grouping_type in disease_list.columns:
            disease_list_copy = disease_list[["id", self.disease_grouping_type]]
            # merge disease list with data
            X_copy = X.copy()
            disease_list_copy = disease_list_copy.loc[~disease_list_copy["id"].duplicated()]
            X_copy = X_copy.merge(disease_list_copy, left_on="target", right_on="id", how="left")
            X_copy = X_copy[~X_copy.id.isna()]
        else:
            raise ValueError(f"Disease grouping type {self.disease_grouping_type} not found in disease_list")

        # get indices of rows where disease type is in holdout_disease_types
        for i in range(self.n_splits):
            selected_disease_types = self.holdout_disease_types[i]

            # We use .str.contains() here to support cases where the disease grouping column
            # contains multiple types in a pipe-separated string (e.g., 'hereditary_disease|metabolic_disease').
            # This ensures that any disease containing the selected type anywhere in the string
            # will be included in the test set for that split.
            mask = (
                X_copy[self.disease_grouping_type]
                .fillna("")
                .str.lower()
                .str.contains(selected_disease_types.lower(), na=False)
            )
            test_indices = X_copy[mask].index.tolist()
            train_indices = X_copy[~mask].index.tolist()
            yield train_indices, test_indices

    def get_n_splits(self, X=None, disease_list=None):
        """Returns the number of splitting iterations in the cross-validator.

        Args:
            X: Ignored, present for API consistency with scikit-learn.
            disease_list: Ignored, present for API consistency with scikit-learn.

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
