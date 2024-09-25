"""Pair generator module.

Module containing drug-disease pair generator
"""
import abc
from tqdm import tqdm
import pandas as pd
import random
from kedro.io import AbstractDataset

from matrix.datasets.graph import KnowledgeGraph

from typing import List, Set, Union


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


## Generators for negative sampling during training


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
            # Sample random drug and disease
            rand_drug = random.choice(drug_samp_ids)
            rand_disease = random.choice(disease_samp_ids)
            # Perform replacements
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


## Generators for evaluation datasets


class GroundTruthTestPairs(DrugDiseasePairGenerator):
    """Class representing ground truth test data."""

    def __init__(
        self, positive_columns: List[str], negative_columns: List[str]
    ) -> None:
        """Initialises an instance of the class.

        Args:
            positive_columns: Names of the flag columns in the matrix which represent the positive pairs.
            negative_columns: Names of the flag columns in the matrix which represent the negative pairs.
        """
        self.positive_columns = positive_columns
        self.negative_columns = negative_columns

    def generate(
        self,
        matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Function to generate the dataset given a full matrix dataframe.

        Args:
            matrix: Pairs dataframe representing the full matrix with treat scores.

        Returns:
            Labelled ground truth drug-disease pairs dataset.
        """
        # Extract and label positive data
        positive_data_lst = []
        for col_name in self.positive_columns:
            positive_data_lst.append(matrix[matrix[col_name]].assign(y=1))

        # Extract and label negative data
        negative_data_lst = []
        for col_name in self.negative_columns:
            negative_data_lst.append(matrix[matrix[col_name]].assign(y=0))

        # Combine data
        data = pd.concat(positive_data_lst + negative_data_lst, ignore_index=True)

        # Return selected pairs
        return data


class MatrixTestDiseases(DrugDiseasePairGenerator):
    """A class representing dataset of pairs required for disease-specific ranking."""

    def __init__(
        self,
        positive_columns: List[str],
        removal_columns: Union[List[str], None] = None,
    ) -> None:
        """Initialises an instance of the class.

        Args:
            positive_columns: Names of the flag columns in the matrix which represent the positive pairs.
            removal_columns: Names of the flag columns in the matrix which represent the pairs to remove.
        """
        self.positive_columns = positive_columns
        self.removal_columns = removal_columns

    def generate(
        self,
        matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate dataset given a full matrix.

        Args:
            matrix: Pairs dataframe representing the full matrix with treat scores.

        Returns:
            Labelled drug-disease pairs dataset.
        """
        # Extract and label positive pairs
        is_positive = pd.Series(False, index=matrix.index)
        for col_name in self.positive_columns:
            is_positive = is_positive | matrix[col_name]
        positive_pairs = matrix[is_positive]
        matrix["y"] = is_positive.astype(int)

        # Restriction to diseases in the positive pairs set
        positive_diseases = positive_pairs["target"].unique()
        in_output = matrix["target"].isin(positive_diseases)

        # Removal of boolean
        if self.removal_columns != None:
            is_remove = pd.Series(False, index=matrix.index)
            for col_name in self.removal_columns:
                is_remove = is_remove | matrix[col_name]
            in_output = in_output | is_remove

        # Apply boolean condition to matrix and return
        return matrix[in_output]


class FullMatrixPositives(DrugDiseasePairGenerator):
    """A class that represents the ranks of a set of positive pairs in a full matrix.

    FUTURE: We may want to add an option to remove selected pairs (e.g. other known positives).
    """

    def __init__(
        self,
        positive_columns: List[str],
    ) -> None:
        """Initialises instance of the class.

        Args:
            positive_columns: Names of the flag columns in the matrix which represent the positive pairs.
        """
        self.positive_columns = positive_columns

    def generate(
        self,
        matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate dataset given a full matrix.

        Args:
            matrix: Pairs dataframe representing the full matrix with treat scores.

        Returns:
            Labelled drug-disease pairs dataset.
        """
        matrix = matrix.reset_index(drop=True)

        # Extract and label positive pairs
        is_positive = pd.Series(False, index=matrix.index)
        for col_name in self.positive_columns:
            is_positive = is_positive | matrix[col_name]
        positive_pairs = matrix[is_positive]

        # Add labels and ranks columns
        positive_pairs["y"] = 1
        positive_pairs["rank"] = positive_pairs.index + 1

        return positive_pairs
