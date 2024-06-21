"""Pair generator module.

Module containing drug-disease pair generator
"""
import abc
import pandas as pd
import random

from matrix.datasets.graph import KnowledgeGraph


class DrugDiseasePairGenerator(abc.ABC):
    """Generator strategy class to represent drug-disease pairs generators."""

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

    def __init__(self, y_label: int, random_state: int, n_unknown: int) -> None:
        """Initializes the RandomDrugDiseasePairGenerator instance.

        Args:
            y_label: label to assign to generated pairs.
            random_state: Random seed.
            n_unknown: Number of unknown drug-disease pairs to generate.
        """
        self._n_unknown = n_unknown
        super().__init__(y_label, random_state)

    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        """Function to generate drug-disease pairs according to the strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with ground truth drug-disease pairs.

        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        # Extract ground truth positive training set
        known_data_set = {
            (drug, disease)
            for drug, disease in zip(known_pairs["source"], known_pairs["target"])
        }

        # Generate unknown data
        unknown_data = []
        while len(unknown_data) < self._n_unknown:
            drug = random.choice(graph._drug_nodes)
            disease = random.choice(graph._disease_nodes)

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

    def __init__(self, y_label: int, random_state: int, n_replacements: int) -> None:
        """Initializes the ReplacementDrugDiseasePairGenerator instance.

        Args:
            y_label: label to assign to generated pairs.
            random_state: Random seed.
            n_replacements: Number of replacements to make.
        """
        self._n_replacements = n_replacements
        super().__init__(y_label, random_state)

    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        """Function to generate drug-disease pairs according to the strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with ground truth drug-disease pairs.

        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        known_data_set = {
            (drug, disease)
            for drug, disease in zip(known_pairs["source"], known_pairs["target"])
        }

        # Extract ground truth positive training set
        kp_train_pairs = known_pairs[
            (known_pairs["y"] == 1) & (known_pairs["split"] == "TRAIN")
        ]
        kp_train_set = {
            (drug, disease)
            for drug, disease in zip(kp_train_pairs["source"], kp_train_pairs["target"])
        }

        # Generate unknown data
        unknown_data = []
        for kp_drug, kp_disease in kp_train_set:
            unknown_data += ReplacementDrugDiseasePairGenerator._make_replacements(
                graph,
                kp_drug,
                kp_disease,
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
        n_replacements: int,
        known_data_set: set[tuple],
        y_label: int,
    ) -> list[str]:
        """Helper function to generate list of drug-disease pairs through replacements."""
        unknown_data = []
        while len(unknown_data) < 2 * n_replacements:
            rand_drug = random.choice(graph._drug_nodes)
            rand_disease = random.choice(graph._disease_nodes)
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
