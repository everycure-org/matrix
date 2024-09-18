"""Pair generator module.

Module containing drug-disease pair generator
"""
import abc
from tqdm import tqdm
import pandas as pd
import random
from kedro.io import AbstractDataset

from matrix.datasets.graph import KnowledgeGraph

from typing import List, Set


class DrugDiseasePairGenerator(abc.ABC):
    """Generator strategy class to represent drug-disease pair generators."""

    @abc.abstractmethod
    def generate(self, known_pairs: pd.DataFrame) -> pd.DataFrame:
        """Function to generate drug-disease pairs from the knowledge graph.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with ground truth drug-disease pairs.

        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        ...

    @staticmethod
    def _remove_pairs(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        """A helper function to remove pairs from a given DataFrame.

        Args:
            df: DataFrame to remove pairs from.
            pairs: DataFrame with pairs to remove.

        Returns:
            DataFrame with train pairs removed.
        """
        pairs_set = set(zip(pairs["source"], pairs["target"]))
        is_remove = df.apply(
            lambda row: (row["source"], row["target"]) in pairs_set, axis=1
        )
        return df[~is_remove]


class SingleLabelPairGenerator(DrugDiseasePairGenerator):
    """Class representing generators outputting drug-disease pairs with a single label."""

    def __init__(self, y_label: int, random_state: int) -> None:
        """Initializes the SingleLabelPairGenerator instance.

        Args:
            y_label: label to assign to generated pairs.
            random_state: Random seed.
        """
        self._y_label = y_label
        self._random_state = random_state
        random.seed(random_state)


class RandomDrugDiseasePairGenerator(SingleLabelPairGenerator):
    """Random drug-disease pair implementation.

    Strategy implementing a drug-disease pair generator using randomly sampled drugs and diseases.

    """

    def __init__(
        self,
        y_label: int,
        random_state: int,
        n_unknown: int,
        drug_flags: List[str],
        disease_flags: List[str],
    ) -> None:
        """Initializes the RandomDrugDiseasePairGenerator instance.

        Args:
            y_label: label to assign to generated pairs.
            random_state: Random seed.
            n_unknown: Number of unknown drug-disease pairs to generate.
            drug_flags: List of knowledge graph flags defining drugs sample set.
            disease_flags: List of knowledge graph flags defining diseases sample set.
        """
        self._n_unknown = n_unknown
        self._drug_flags = drug_flags
        self._disease_flags = disease_flags
        super().__init__(y_label, random_state)

    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Function to generate drug-disease pairs according to the strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.
            kwargs: additional kwargs to use
        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        # Define ground truth dataset
        known_data_set = {
            (drug, disease)
            for drug, disease in zip(known_pairs["source"], known_pairs["target"])
        }

        # Defining list of node id's to sample from
        drug_samp_ids = graph.flags_to_ids(self._drug_flags)
        disease_samp_ids = graph.flags_to_ids(self._disease_flags)

        # Sample pairs
        unknown_data = []
        while len(unknown_data) < self._n_unknown:
            drug = random.choice(drug_samp_ids)
            disease = random.choice(disease_samp_ids)

            if (drug, disease) not in known_data_set:
                unknown_data.append(
                    [
                        drug,
                        graph._embeddings[drug],
                        disease,
                        graph._embeddings[disease],
                        self._y_label,
                    ]
                )

        return pd.DataFrame(
            columns=["source", "source_embedding", "target", "target_embedding", "y"],
            data=unknown_data,
        )


class ReplacementDrugDiseasePairGenerator(SingleLabelPairGenerator):
    """Replacement drug-disease pair implementation.

    Strategy implementing a drug-disease pair generator using random drug and disease replacements.

    """

    def __init__(
        self,
        y_label: int,
        random_state: int,
        n_replacements: int,
        drug_flags: List[str],
        disease_flags: List[str],
    ) -> None:
        """Initializes the ReplacementDrugDiseasePairGenerator instance.

        Args:
            y_label: label to assign to generated pairs.
            random_state: Random seed.
            n_replacements: Number of replacements to make.
            drug_flags: List of knowledge graph flags defining drugs sample set.
            disease_flags: List of knowledge graph flags defining diseases sample set.
        """
        self._n_replacements = n_replacements
        self._drug_flags = drug_flags
        self._disease_flags = disease_flags
        super().__init__(y_label, random_state)

    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Function to generate drug-disease pairs according to the strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.
            kwargs: additional kwargs to use
        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        known_data_set = {
            (drug, disease)
            for drug, disease in zip(known_pairs["source"], known_pairs["target"])
        }

        # Extract known positive training set
        kp_train_pairs = known_pairs[
            (known_pairs["y"] == 1) & (known_pairs["split"] == "TRAIN")
        ]
        kp_train_set = {
            (drug, disease)
            for drug, disease in zip(kp_train_pairs["source"], kp_train_pairs["target"])
        }
        # Defining list of node id's to sample from
        drug_samp_ids = graph.flags_to_ids(self._drug_flags)
        disease_samp_ids = graph.flags_to_ids(self._disease_flags)

        # Generate unknown data
        unknown_data = []
        for kp_drug, kp_disease in tqdm(kp_train_set):
            unknown_data += ReplacementDrugDiseasePairGenerator._make_replacements(
                graph,
                kp_drug,
                kp_disease,
                drug_samp_ids,
                disease_samp_ids,
                self._n_replacements,
                known_data_set,
                self._y_label,
            )

        return pd.DataFrame(
            columns=["source", "source_embedding", "target", "target_embedding", "y"],
            data=unknown_data,
        )

    @staticmethod
    def _make_replacements(
        graph: KnowledgeGraph,
        kp_drug: str,
        kp_disease: str,
        drug_samp_ids: List[str],
        disease_samp_ids: List[str],
        n_replacements: int,
        known_data_set: Set[tuple],
        y_label: int,
    ) -> List[str]:
        """Helper function to generate list of drug-disease pairs through replacements."""
        # Sample pairs
        unknown_data = []
        while len(unknown_data) < 2 * n_replacements:
            rand_drug = random.choice(drug_samp_ids)
            rand_disease = random.choice(disease_samp_ids)
            if (kp_drug, rand_disease) not in known_data_set and (
                rand_drug,
                kp_disease,
            ) not in known_data_set:
                for drug, disease in [
                    (kp_drug, rand_disease),
                    (rand_drug, kp_disease),
                ]:
                    unknown_data.append(
                        [
                            drug,
                            graph._embeddings[drug],
                            disease,
                            graph._embeddings[disease],
                            y_label,
                        ]
                    )
        return unknown_data


class GroundTruthTestPairs(DrugDiseasePairGenerator):
    """Class representing ground truth test data."""

    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Function generating ground truth pairs.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: Labelled ground truth drug-disease pairs dataset.
            kwargs: additional kwargs to use
        Returns:
            Labelled ground truth drug-disease pairs dataset.
        """
        # Restrict to test portion
        is_test = known_pairs["split"].eq("TEST")
        test_pairs = known_pairs[is_test]

        # Return selected pairs
        return test_pairs


class MatrixTestDiseases(DrugDiseasePairGenerator):
    """A class representing the test diseases x all drugs matrix.

    This dataset consists of drug-disease pairs obtained by taking all combinations of:
        - drugs in a given list,
        - diseases appearing in the ground-truth positive test dataset,
    while omitting any ground-truth training data.
    """

    def __init__(self, drugs_lst_flags: List[str]) -> None:
        """Initializes the MatrixTestDiseases instance.

        Args:
            drugs_lst_flags: List of knowledge graph flags defining a list of drugs.
        """
        self._drug_axis_flags = drugs_lst_flags

    def _give_disease_centric_matrix(
        self,
        test_pos_pairs: pd.DataFrame,
        removal_pairs: pd.DataFrame,
        drug_list: list,
    ) -> pd.DataFrame:
        """Generate disease-centric matrix.

        The disease-centric matrix is defined as the set of drug disease-pairs for which
        the drug belongs to a given list and the disease appears in the ground truth positive test set.
        We remove certain pairs such as the training set.
        We label the ground truth test positives by y=1 and everything else as y=0.

        Args:
            test_pos_pairs: _description_
            removal_pairs: Drug-disease pairs to remove.
            drug_list: List of node IDs representing the drugs.

        Returns:
            Labelled drug-disease pairs dataset.
        """
        # Compute list of disease IDs
        test_pos_diseases_lst = list(test_pos_pairs["target"].unique())

        # Generate all combinations
        matrix_slices = []
        for disease in tqdm(test_pos_diseases_lst):
            matrix_slice = pd.DataFrame({"source": drug_list, "target": disease})
            matrix_slices.append(matrix_slice)

        # Concatenate all slices at once
        matrix = pd.concat(matrix_slices, ignore_index=True)

        # Label test positives
        test_pos_pairs_set = set(
            zip(test_pos_pairs["source"], test_pos_pairs["target"])
        )
        is_in_test_pos = matrix.apply(
            lambda row: (row["source"], row["target"]) in test_pos_pairs_set, axis=1
        )
        matrix["y"] = is_in_test_pos.astype(int)

        # Remove train pairs
        filtered_matrix = self._remove_pairs(matrix, removal_pairs)

        return filtered_matrix

    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Function generating the test diseases x all drugs matrix dataset.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: Labelled ground truth drug-disease pairs dataset.
            kwargs: additional kwargs to use
        Returns:
            Labelled drug-disease pairs dataset.
        """
        # Separate test and train portions of ground truth
        is_test = known_pairs["split"].eq("TEST")
        test_pairs = known_pairs[is_test]
        train_pairs = known_pairs[~is_test]

        # Define lists of drugs and diseases
        test_pos_pairs = test_pairs[test_pairs["y"].eq(1)]
        drug_list = graph.flags_to_ids(self._drug_axis_flags)

        return self._give_disease_centric_matrix(test_pos_pairs, train_pairs, drug_list)


class TimeSplitGroundTruthTestPairs(DrugDiseasePairGenerator):
    """Data Generator for Time Split Validation. Use the clinical trial data to replace the test ground truth data.

    Now 1 in the 'y' column means 'significantly_better' and 0 means 'significantly_worse'.
    The ground truth training data is removed.

    """

    def generate(
        self,
        graph: KnowledgeGraph,
        known_pairs: pd.DataFrame,
        clinical_trials_data: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Function to generate drug-disease pairs from the knowledge graph.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with ground truth drug-disease pairs.
            clinical_trials_data: clinical trails dataset
            kwargs: additional kwargs to use
        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        # Extract the known DD ground truth used in model training
        is_test = known_pairs["split"].eq("TEST")
        train_pairs = known_pairs[~is_test]

        # Remove train pairs from the clinical trail data
        clinical_trial_data = self._remove_pairs(clinical_trials_data, train_pairs)

        # Check if column 'y' has both 0 and 1 values
        if clinical_trial_data["y"].nunique() != 2:
            raise ValueError("Column 'y' should have both 0 and 1 values.")
        else:
            return clinical_trial_data


class TimeSplitMatrixTestDiseases(MatrixTestDiseases):
    """Data Generator for Time Split Validation. Use the clinical trial data to replace the test ground truth data.

    The matrix pairs dataset  consists of diseases of clinical trial data with "significant_better" label x all drugs.
    Now 1 in the 'y' column means 'significantly_better' and 0 is everything else.
    All ground truth data (both test and train) is removed.

    TODO: Consider expanding pairs labelled as y=1 to include also "non-significantly better" pairs.
    """

    def __init__(self, drugs_lst_flags: str) -> None:
        """Initializes the SingleLabelPairGenerator instance.

        Args:
            drugs_lst_flags: List of knowledge graph flags defining drugs sample set.
        """
        self._drug_axis_flags = drugs_lst_flags

    def generate(
        self,
        graph: KnowledgeGraph,
        known_pairs: pd.DataFrame,
        clinical_trials_data: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Function to generate drug-disease pairs from the knowledge graph.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with ground truth drug-disease pairs.
            clinical_trials_data: clinical trails dataset
            kwargs: additional kwargs to use
        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        # Define lists of drugs and diseases
        clinical_trial_data_pos_pairs = clinical_trials_data[
            clinical_trials_data["y"].eq(1)
        ]
        drug_list = graph.flags_to_ids(self._drug_axis_flags)

        # Generate matrix dataset
        return self._give_disease_centric_matrix(
            clinical_trial_data_pos_pairs, known_pairs, drug_list
        )
