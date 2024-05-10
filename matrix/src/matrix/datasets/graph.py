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

    def __init__(self, nodes: pd.DataFrame) -> None:
        self._nodes = nodes
        self._node_index = dict(zip(nodes['id'], nodes.index))

        # Add type specific indexes
        self._drug_nodes = list(nodes[nodes['is_drug']]["id"])
        self._disease_nodes = list(nodes[nodes['is_disease']]["id"])
    

class DrugDiseasePairGenerator(abc.ABC):

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
        pass


class RandomDrugDiseasePairGenerator(DrugDiseasePairGenerator):

    def generate(
        self, 
        graph: KnowledgeGraph, 
        known_pairs: pd.DataFrame, 
        n_unknown: int
    ) -> pd.DataFrame:

        known_data_set = {(drug, disease) for drug, disease in zip(known_pairs['source'], known_pairs['target'])}

        unknown_data = []
        while len(unknown_data) < n_unknown:
            
            drug = random.choice(graph._drug_nodes)
            disease = random.choice(graph._disease_nodes)

            if (drug, disease) not in known_data_set:
                unknown_data.append([drug, disease, 2])
            
        return pd.DataFrame(columns=['source', 'target', 'y'], data=unknown_data)
        


class KnowledgeGraphDataset(ParquetDataset):

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