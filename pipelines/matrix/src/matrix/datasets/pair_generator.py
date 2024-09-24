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

        TODO: Remove.

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

    @staticmethod
    def _check_no_train(data: pd.DataFrame, known_pairs: pd.DataFrame) -> None:
        """Checks that no pairs in the ground truth training set appear in the data.

        TODO: This could take a long time for large dataframes. Move this to a node that checks each matrix.

        Args:
            data: Pairs dataset to check.
            known_pairs: DataFrame with known drug-disease pairs.

        Raises:
            ValueError: If any training pairs are found in the data.
        """
        is_test = known_pairs["split"].eq("TEST")
        train_pairs = known_pairs[~is_test]
        train_pairs_set = set(zip(train_pairs["source"], train_pairs["target"]))
        data_pairs_set = set(zip(data["source"], data["target"]))
        overlapping_pairs = data_pairs_set.intersection(train_pairs_set)
        if overlapping_pairs:
            raise ValueError(
                f"Found {len(overlapping_pairs)} pairs in test set that also appear in training set."
            )


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

    def generate(
        self,
        known_pairs: pd.DataFrame,
        matrix: pd.DataFrame,
        eval_options: dict,
    ) -> pd.DataFrame:
        """Function to generate the dataset given a full matrix dataframe.

        Args:
            known_pairs: Labelled ground truth drug-disease pairs dataset.
            matrix: Pairs dataframe representing the full matrix with treat scores.
            eval_options: Dictionary of parameters containing lists of column names for positive/negative data.

        Returns:
            Labelled ground truth drug-disease pairs dataset.
        """
        # Extract and label positive data
        positive_data_lst = []
        for col_name in eval_options["positive_columns"]:
            positive_data_lst.append(matrix[matrix[col_name]].assign(y=1))

        # Extract and label negative data
        negative_data_lst = []
        for col_name in eval_options["negative_columns"]:
            negative_data_lst.append(matrix[matrix[col_name]].assign(y=0))

        # Combine data
        data = pd.concat(positive_data_lst + negative_data_lst, ignore_index=True)

        # Check that ground truth training pairs do not appear in the test set
        self._check_no_train(data, known_pairs)

        # Return selected pairs
        return data


class MatrixTestDiseases(DrugDiseasePairGenerator):
    """A class representing dataset of pairs required for disease-specific ranking."""

    def generate(
        self,
        matrix: pd.DataFrame,
        positive_columns_lst: List[str],
        removal_columns_lst: List[str],
    ) -> pd.DataFrame:
        """Generate dataset given a full matrix.

        Args:
            matrix: Pairs dataframe representing the full matrix with treat scores.
            positive_columns_lst: List of column names defining the "positive" pairs labelled as y=1.
            removal_columns_lst: Drug-disease pairs to remove.

        Returns:
            Labelled drug-disease pairs dataset.
        """
        # Restrict diseases boolean

        # Removal boolean

        # Combine boolean, restrict boolean and label pairs.

        return None
