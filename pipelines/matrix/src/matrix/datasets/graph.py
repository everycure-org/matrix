import pandas as pd
import abc
import random
import itertools

from kedro_datasets.pandas import CSVDataset
from kedro_datasets.pandas import ParquetDataset

from typing import Any, Dict
from kedro.io.core import (
    PROTOCOL_DELIMITER,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)

class KnowledgeGraph:
    """
    Class to represent a knowledge graph.

    NOTE: Provide handover point to Neo4J in the future.
    """

    def __init__(self, nodes: pd.DataFrame) -> None:
        self._nodes = nodes
        self._node_index = dict(zip(nodes['id'], nodes.index))

        # Add type specific indexes
        self._drug_nodes = list(nodes[nodes['is_drug']]["id"])
        self._disease_nodes = list(nodes[nodes['is_disease']]["id"])
    

class DrugDiseasePairGenerator(abc.ABC):
    """
    Generator strategy class to represent drug-disease pairs generators.
    """

    def __init__(self, random_state: int) -> None:
        self._random_state = random_state
        random.seed(random_state)

    @abc.abstractmethod
    def generate(
        self, 
        graph: KnowledgeGraph, 
        known_pairs: pd.DataFrame,
        n_unknown: int
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


class RandomDrugDiseasePairGenerator(DrugDiseasePairGenerator):

    def generate(
        self, 
        graph: KnowledgeGraph, 
        known_pairs: pd.DataFrame, 
        n_unknown: int
    ) -> pd.DataFrame:
        """
        Function to generate drug-disease pairs using a randomized strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.
            n_unknown: Number of unknown drug-disease pairs to generate.
        Returns:
            DataFrame with unknown drug-disease pairs.
        """

        known_data_set = {(drug, disease) for drug, disease in zip(known_pairs['source'], known_pairs['target'])}

        unknown_data = []
        while len(unknown_data) < n_unknown:
            
            drug = random.choice(graph._drug_nodes)
            disease = random.choice(graph._disease_nodes)

            if (drug, disease) not in known_data_set:
                unknown_data.append([drug, disease, 2])
            
        return pd.DataFrame(columns=['source', 'target', 'y'], data=unknown_data)
        
# TODO: Alexei add negative class and try to run pipeline with it

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
            metadata=metadata
        )


    def _load(self) -> KnowledgeGraph:    
        return KnowledgeGraph(super()._load())