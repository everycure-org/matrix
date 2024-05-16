import pandas as pd
import abc
import random

from kedro_datasets.pandas import ParquetDataset

from typing import Any, Dict
from kedro.io.core import Version


class KnowledgeGraph:
    """
    Class to represent a knowledge graph.

    NOTE: Provide handover point to Neo4J in the future.
    """

    def __init__(self, nodes: pd.DataFrame) -> None:
        self._nodes = nodes
        self._node_index = dict(zip(nodes["id"], nodes.index))

        # Add type specific indexes
        self._drug_nodes = list(nodes[nodes["is_drug"]]["id"])
        self._disease_nodes = list(nodes[nodes["is_disease"]]["id"])
        self._embeddings = dict(zip(nodes["id"], nodes["embedding"]))


class DrugDiseasePairGenerator(abc.ABC):
    """
    Generator strategy class to represent drug-disease pairs generators.
    """

    def __init__(self, random_state: int) -> None:
        self._random_state = random_state
        random.seed(random_state)

    @abc.abstractmethod
    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame, n_unknown: int
    ) -> pd.DataFrame:
        """
        Function to generate drug-disease pairs from the knowledge graph.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.
            n_unknown: Number of unknown drug-disease pairs to generate.
        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        ...


class JointDrugDiseasePairGenerator(DrugDiseasePairGenerator):
    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame, n_unknown: int
    ) -> pd.DataFrame:
        """
        Function to generate drug-disease pairs using a joint distribution strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.
            n_unknown: Number of unknown drug-disease pairs to generate.
                        
        Returns:
            DataFrame with unknown drug-disease pairs.
        """

        known_data_set = {
            (drug, disease)
            for drug, disease in zip(known_pairs["source"], known_pairs["target"])
        }

        unknown_data = []
        while len(unknown_data) < n_unknown:
            drug = random.choice(graph._drug_nodes)
            disease = random.choice(graph._disease_nodes)

            if (drug, disease) not in known_data_set:
                unknown_data.append(
                    [
                        drug,
                        graph._embeddings[drug],
                        disease,
                        graph._embeddings[disease],
                        2,
                    ]
                )

        return pd.DataFrame(
            columns=["source", "source_embedding", "target", "target_embedding", "y"],
            data=unknown_data,
        )


class NegativeDrugDiseasePairGenerator(DrugDiseasePairGenerator):
    def generate(
        self, graph: KnowledgeGraph, known_pairs: pd.DataFrame, n_unknown: int
    ) -> pd.DataFrame:
        """
        Function to generate drug-disease pairs using a randomized strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.
            n_unknown: Number of unknown drug-disease pairs to generate. If n_unknown is odd,
                    number of generated pairs will be rounded to nearest even number 
        Returns:
            DataFrame with unknown drug-disease pairs.
        """

        known_data_set = {
            (drug, disease)
            for drug, disease in zip(known_pairs["source"], known_pairs["target"])
        }

        # Extract known positive training set
        train_pairs = known_pairs[known_pairs['split']=='TRAIN']
        kp_train_pairs = train_pairs[train_pairs['y']==1]
        kp_train_lst = [
            (drug, disease)
            for drug, disease in zip(kp_train_pairs["source"], kp_train_pairs["target"])
        ]

        # Generate unknown data
        unknown_data = []
        idx = 0
        while len(unknown_data) < n_unknown:
            if idx < len(kp_train_lst):
                kp_drug, kp_disease = kp_train_lst[idx]
            else:
                raise IndexError("Not enough KP training data")
            
            rand_drug = random.choice(graph._drug_nodes)
            rand_disease = random.choice(graph._disease_nodes)

            if (kp_drug, rand_disease) not in known_data_set and (rand_drug , kp_disease) not in known_data_set:
                for drug, disease in [(kp_drug, rand_disease), (rand_drug, kp_disease)]:
                    unknown_data.append(
                        [
                            drug,
                            graph._embeddings[drug],
                            disease,
                            graph._embeddings[disease],
                            2,
                        ]
                    )
                    idx += 1

        return pd.DataFrame(
            columns=["source", "source_embedding", "target", "target_embedding", "y"],
            data=unknown_data,
        )
    

class KnowledgeGraphDataset(ParquetDataset):
    """
    Dataset adaptor to read KnowledgeGraph using Kedro's dataset functionality.
    """

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
