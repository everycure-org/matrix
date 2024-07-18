"""Pair generator module.

Module containing drug-disease pair generator
"""
import abc
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import logging

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
    def __init__(
        self, drug_flags, disease_flags, n_replacements, y_label: int, random_state: int
    ):
        super().__init__(y_label, random_state)
        self._drug_flags = drug_flags
        self._disease_flags = disease_flags
        self._n_replacements = n_replacements
        self._rng = np.random.default_rng(random_state)

    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        print(f"1) get drugs and diseases lists")
        drug_ids = np.array(graph.flags_to_ids(self._drug_flags))
        disease_ids = np.array(graph.flags_to_ids(self._disease_flags))
        print(f"1) {len(drug_ids)} drugs and {len(disease_ids)} diseases lists")

        # Convert known pairs to a set for fast lookup
        known_pairs_set = set(map(tuple, known_pairs[["source", "target"]].values))

        print("2) create maps of embeddings")
        drug_embeddings = {drug: graph._embeddings[drug] for drug in drug_ids}
        disease_embeddings = {
            disease: graph._embeddings[disease] for disease in disease_ids
        }

        print("3) how many replacements do we need? ")
        kp_train_pairs = known_pairs[
            (known_pairs["y"] == 1) & (known_pairs["split"] == "TRAIN")
        ]
        total_replacements = len(kp_train_pairs) * self._n_replacements * 2

        # Generate pairs in chunks
        chunk_size = 1_000_000  # Adjust this based on available memory
        generated_pairs = []

        print("4) Iterate in chunks")
        while len(generated_pairs) < total_replacements:
            # Generate a chunk of random pairs
            drug_chunk = self._rng.choice(drug_ids, chunk_size)
            disease_chunk = self._rng.choice(disease_ids, chunk_size)
            pairs_chunk = np.column_stack((drug_chunk, disease_chunk))

            # Filter out known pairs
            mask = np.array(
                [tuple(pair) not in known_pairs_set for pair in pairs_chunk]
            )
            new_pairs = pairs_chunk[mask]

            # Add new pairs to the result
            generated_pairs.extend(new_pairs)

            # Break if we have enough pairs
            if len(generated_pairs) >= total_replacements:
                generated_pairs = generated_pairs[:total_replacements]
                break

        # Create result DataFrame
        result = pd.DataFrame(generated_pairs, columns=["source", "target"])
        result["source_embedding"] = result["source"].map(drug_embeddings)
        result["target_embedding"] = result["target"].map(disease_embeddings)
        result["y"] = self._y_label

        return result

    def _generate_pairs(
        self,
        graph,
        kp_train_set,
        known_data_set,
        drug_ids,
        disease_ids,
        drug_embeddings,
        disease_embeddings,
    ):
        unknown_data = []
        batch_size = 1000  # Adjust this based on your memory constraints

        for kp_drug, kp_disease in tqdm(kp_train_set):
            pairs_needed = 2 * self._n_replacements
            while pairs_needed > 0:
                batch = min(batch_size, pairs_needed)

                rand_drugs = np.random.choice(drug_ids, batch)
                rand_diseases = np.random.choice(disease_ids, batch)

                new_pairs = set(zip([kp_drug] * batch, rand_diseases)) | set(
                    zip(rand_drugs, [kp_disease] * batch)
                )
                new_pairs -= known_data_set

                for drug, disease in new_pairs:
                    unknown_data.append(
                        [
                            drug,
                            graph._embeddings[drug],
                            disease,
                            graph._embeddings[disease],
                            self._y_label,
                        ]
                    )
                    pairs_needed -= 1
                    if pairs_needed == 0:
                        break

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
        for idx, disease in enumerate(test_diseases_lst):
            matrix_slice = pd.DataFrame({"source": drugs_lst, "target": disease})
            test_pos_pairs_in_slice = test_pairs[test_pairs["target"].eq(disease)]
            test_pos_drugs_in_slice = test_pos_pairs_in_slice["source"]
            is_test_pos = matrix_slice["source"].isin(test_pos_drugs_in_slice)
            matrix_slice["y"] = is_test_pos.astype(int)
            if idx == 0:
                matrix = matrix_slice
            else:
                matrix = pd.concat([matrix, matrix_slice], ignore_index=True)

        # Remove training data
        are_pairs_equal = lambda pair_1, pair_2: (
            pair_1["source"] == pair_2["source"]
        ) and (pair_1["target"] == pair_2["target"])
        is_pair_in_train = lambda pair: pd.Series(
            [
                are_pairs_equal(pair, train_pair)
                for _, train_pair in train_pairs.iterrows()
            ]
        ).any()
        is_in_train = matrix.apply(is_pair_in_train, axis=1)
        return matrix[~is_in_train]
