"""Graph module.

Module containing knowledge graph representation and utilities.
"""
import pandas as pd
import abc
import random

from kedro_datasets.pandas import ParquetDataset

from typing import Any, Dict, Iterator, List, Set
from kedro.io.core import Version


class KnowledgeGraph:
    """Class to represent a knowledge graph.

    NOTE: Provide handover point to Neo4J in the future.
    """

    def __init__(self, nodes: pd.DataFrame) -> None:
        """Initializes the KnowledgeGraph instance.

        Args:
            nodes: DataFrame containing nodes of the graph.
        """
        self._nodes = nodes
        self._node_index = dict(zip(nodes["id"], nodes.index))

        # Add type specific indexes
        self._drug_nodes = list(nodes[nodes["is_drug"]]["id"])
        self._disease_nodes = list(nodes[nodes["is_disease"]]["id"])
        self._embeddings = dict(zip(nodes["id"], nodes["embedding"]))


class DrugDiseasePairGenerator(abc.ABC):
    """Generator strategy class to represent drug-disease pairs generators."""

    def __init__(self, y_label: int, random_state: int) -> None:
        """Initializes the DrugDiseasePairGenerator instance.

        Args:
            y_label: label to assign to generated pairs.
            random_state: Random seed.
        """
        self._y_label = y_label
        self._random_state = random_state
        random.seed(random_state)

    @abc.abstractmethod
    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        """Function to generate drug-disease pairs from the knowledge graph.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.

        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        ...


def flags_to_ids(graph: KnowledgeGraph, flags: List[str]) -> List[str]:
    """Helper function for extracting nodes from flag columns.

    Args:
        graph: KnowledgeGraph instance.
        flags: List of names of boolean columns in the graph nodes.

    Returns:
        List of graph nodes id's satisfying all flags.
    """
    is_all_flags = graph._nodes[flags].all(axis=1)
    select_nodes = graph._nodes[is_all_flags]
    return list(select_nodes["id"])


class RandomDrugDiseasePairGenerator(DrugDiseasePairGenerator):
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
        drug_samp_ids = flags_to_ids(graph, self._drug_flags)
        disease_samp_ids = flags_to_ids(graph, self._disease_flags)

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


class ReplacementDrugDiseasePairGenerator(DrugDiseasePairGenerator):
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

        # Generate unknown data
        unknown_data = []
        for kp_drug, kp_disease in kp_train_set:
            unknown_data += ReplacementDrugDiseasePairGenerator._make_replacements(
                graph,
                kp_drug,
                kp_disease,
                self._drug_flags,
                self._disease_flags,
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
        drug_flags: List[str],
        disease_flags: List[str],
        n_replacements: int,
        known_data_set: Set[tuple],
        y_label: int,
    ) -> List[str]:
        """Helper function to generate list of drug-disease pairs through replacements."""
        # Defining list of node id's to sample from
        drug_samp_ids = flags_to_ids(graph, drug_flags)
        disease_samp_ids = flags_to_ids(graph, disease_flags)

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


class KnowledgeGraphDataset(ParquetDataset):
    """Dataset adaptor to read KnowledgeGraph using Kedro's dataset functionality."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        version: Version = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Initializes the KnowledgeGraphDataset."""
        super().__init__(
            filepath=filepath,
            load_args=load_args,
            save_args=save_args,
            version=version,
            credentials=credentials,
            fs_args=fs_args,
            metadata=metadata,
        )

    def _load(self) -> KnowledgeGraph:
        return KnowledgeGraph(super()._load())
