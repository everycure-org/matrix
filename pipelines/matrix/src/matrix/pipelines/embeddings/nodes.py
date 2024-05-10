from typing import List
import pandas as pd

from sklearn.model_selection._split import _BaseKFold

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema

from matrix.datasets.graph import KnowledgeGraph, DrugDiseasePairGenerator

def create_prm_pairs(
    raw_tp: pd.DataFrame,
    raw_tn: pd.DataFrame
) -> pd.DataFrame:
    
    # Add label
    raw_tp['y'] = 1
    raw_tn['y'] = 0

    # Return concatenated data
    return pd.concat([raw_tp, raw_tn], axis=0).reset_index(drop=True)


@has_schema(
    schema={
        "is_drug": "bool",
        "is_disease": "bool",
        "is_fda_approved": "bool"
    },
    allow_subset=True
)
def create_feat_nodes(
    raw_nodes: pd.DataFrame,
    drug_types: List[str],
    disease_types: List[str],
    fda_list: List[str],
) -> pd.DataFrame:
    """
    Add features for nodes.

    Args:
        raw_nodes: Raw nodes data.
        drug_types: List of drug types.
        disease_types: List of disease types.
        fda_list: List of FDA approved drugs. 
    """
    
    raw_nodes["is_drug"] = raw_nodes["category"].apply(lambda x: x in drug_types)
    raw_nodes["is_disease"] = raw_nodes["category"].apply(lambda x: x in disease_types)
    raw_nodes["is_fda_approved"] = raw_nodes["id"].apply(lambda x: x in fda_list)

    return raw_nodes

@inject_object()
def create_model_input_nodes(
    graph: KnowledgeGraph,
    known_pairs: pd.DataFrame,
    generator: DrugDiseasePairGenerator
) -> pd.DataFrame:
    
    return generator.generate(graph, known_pairs, n_unknown=50)


@inject_object()
def make_splits(
    data: pd.DataFrame,
    splitter: _BaseKFold,
) -> pd.DataFrame:
    
    all_data_frames = []
    for iteration, (train_index, test_index) in enumerate(splitter.split(data, data["y"])):
        all_indices_in_this_fold = list(set(train_index).union(test_index))
        fold_data = data.loc[all_indices_in_this_fold, :].copy()
        fold_data.loc[:, "iteration"] = iteration
        fold_data.loc[train_index, "split"] = "TRAIN"
        fold_data.loc[test_index, "split"] = "TEST"
        all_data_frames.append(fold_data)

    return pd.concat(all_data_frames, axis=0, ignore_index=True)


