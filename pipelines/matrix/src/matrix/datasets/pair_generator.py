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
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        """Function to generate drug-disease pairs according to the strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.

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
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        """Function to generate drug-disease pairs according to the strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.

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
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        """Function generating ground truth pairs.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: Labelled ground truth drug-disease pairs dataset.

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

    def __init__(self, drugs_lst_flags: str) -> None:
        """Initializes the MatrixTestDiseases instance.

        Args:
            drugs_lst_flags: List of knowledge graph flags defining a list of drugs.
        """
        self._drug_axis_flags = drugs_lst_flags

    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        """Function generating the test diseases x all drugs matrix dataset.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: Labelled ground truth drug-disease pairs dataset.

        Returns:
            Labelled drug-disease pairs dataset.
        """
        # Separate test and train portions of ground truth
        is_test = known_pairs["split"].eq("TEST")
        test_pairs = known_pairs[is_test]
        train_pairs = known_pairs[~is_test]

        # Define lists of  drugs and diseases
        test_diseases_lst = list(test_pairs["target"].unique())
        drugs_lst = graph.flags_to_ids(self._drug_axis_flags)

        # Generate all combinations
        matrix_slices = []
        for disease in tqdm(test_diseases_lst):
            matrix_slice = pd.DataFrame({"source": drugs_lst, "target": disease})
            test_pos_pairs_in_slice = test_pairs[test_pairs["target"].eq(disease)]
            test_pos_drugs_in_slice = test_pos_pairs_in_slice["source"]
            is_test_pos = matrix_slice["source"].isin(test_pos_drugs_in_slice)
            matrix_slice["y"] = is_test_pos.astype(int)
            matrix_slices.append(matrix_slice)

        # Concatenate all slices at once
        matrix = pd.concat(matrix_slices, ignore_index=True)
        # Create a set of tuples from train_pairs for faster lookup
        train_pairs_set = set(zip(train_pairs["source"], train_pairs["target"]))

        # Replace is_pair_in_train function with a set lookup
        is_in_train = matrix.apply(
            lambda row: (row["source"], row["target"]) in train_pairs_set, axis=1
        )

        filtered_matrix = matrix[~is_in_train]
        return filtered_matrix


class GeneratorWithSideInput:
    """Generator with dataset input."""

    def __init__(self, dataset: AbstractDataset) -> None:
        """Initializes the SingleLabelPairGenerator instance.

        Args:
            dataset: dataset to use
        """
        # NOTE: I'm fine with the rudimentary data cleaning happening in here for now, is we mix
        # this injection mechanism with ordinary datasets, it may break lineage of the pipeline.
        # there are ways to fix that again, but let's keep it simple. So proposing to add
        # the a clean_data() method to this class and turn the class into TimeSplitValidationDataGenerator?
        self._clinical_trail_data = dataset._load()

    def generate(self, known_pairs: pd.DataFrame) -> pd.DataFrame:
        """Function to generate drug-disease pairs from the knowledge graph.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with ground truth drug-disease pairs.

        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        # Add logic here, will have access to _data to curate dataset you'd like
        breakpoint()
