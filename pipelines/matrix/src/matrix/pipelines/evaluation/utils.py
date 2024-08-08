
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn as skl
import bisect
from typing import Any, List, Dict, Union, Tuple

import sklearn.metrics as skl
from sklearn.impute._base import _BaseImputer

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.unpack import unpack_params
from refit.v1.core.make_list_regexable import _extract_elements_in_list

from matrix.datasets.graph import KnowledgeGraph
from matrix.pipelines.modelling.nodes import _predict_scores
from matrix.pipelines.modelling.model import ModelWrapper


@inject_object()
def perform_disease_centric_evaluation(
            graph: KnowledgeGraph,
            model: ModelWrapper,
            transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
            test_data: pd.DataFrame,
            train_data: pd.DataFrame,
            drug_nodes: pd.DataFrame,
            k_lst: List[int],
            is_return_curves = False):
    """
    Gives "disease-centric" AUROC, AP, MRR and Hit@k scores. Optionally, the ROC and precision-recall curves are given as well.
    For the computation of these scores, the "positive" test dataset is given and the "negative" dataset is constructed by pairing 
    the set of diseases in the positive test dataset with any other drugs not appearing in the positive test dataset , nor the training set.

    Args:
        - graph: KnowledgeGraph object containing the nodes of the graph
        - model: ModelWrapper object containing the model
        - transformers: Dictionary containing the transformers used to transform the input data
        - test_data: DataFrame containing the test data
        - train_data: DataFrame containing the training data
        - drug_nodes: DataFrame containing the drug nodes
        - k_lst: List of integers containing the values of k for which to compute Hit@k scores
        - is_return_curves: Boolean indicating whether to return the ROC and PR curves or not
    Returns:
        - (list) If is_return_curves = False, returns a list with AUROC, AP and MRR scores (in that order), followed by the Hit@k scores.
        - (tuple) Returns a list as above, followed by tuples of vectors reresenting the ROC and PR curves (see skl.metrics.roc_curve and 
        skl.metrics.precision_recall_curve) 
    """
    
    # Re-name columns if necessary
    if 'source' in test_data.columns:
        test_data = test_data.rename(columns = {'source':'drug_kg_id', 'target':'disease_kg_id'})
        test_data = test_data[['drug_kg_id', 'disease_kg_id']]
        test_data['y'] = 1
    if 'source' in train_data.columns:
        train_data = train_data.rename(columns = {'source':'drug_kg_id', 'target':'disease_kg_id'})
    
    # Prepare some variables
    all_drugs = drug_nodes
    test_diseases = pd.Series(test_data['disease_kg_id'].unique())
    mrr_total = 0
    hitk_total_lst = np.zeros(len(k_lst))
    df_all_exists = False # To avoid concatenating empty dataframe while constructing df_all

    # Generate predicted scores for the test data
    test_data = _predict_scores(graph, model, transformers, test_data)

    for disease in tqdm(list(test_diseases)):
        # Extract relevant train and test datapoints
        all_pos_test = test_data[test_data['disease_kg_id']==disease]
        
        all_train = train_data[train_data['disease_kg_id']==disease]
        
        # Construct negative pairs
        check_cond_pos_test =  lambda drug: drug not in list(all_pos_test['drug_kg_id'])
        check_cond_train =  lambda drug: drug not in list(all_train['drug_kg_id'])
        check_conds = lambda drug: check_cond_pos_test(drug) and check_cond_train(drug)
        negative_drugs = all_drugs[all_drugs.map(check_conds)]
        negative_pairs = pd.DataFrame({'source':negative_drugs, 'target':disease})
        negative_pairs['y'] = 0
        
        # Get 'treat' probability scores 
        all_pos_test_probs = all_pos_test['treat score'].to_list()

        # Compute probabilities for negative pairs
        negative_pairs = _predict_scores(graph, model, transformers, negative_pairs)

        # Concatenate to DataFrame with all probability scores
        if df_all_exists:
            df_all = pd.concat((df_all, all_pos_test, negative_pairs), ignore_index=True)
        else: 
            df_all = pd.concat((all_pos_test, negative_pairs), ignore_index=True)
            df_all_exists = True

        # Compute rank for all positives
        negative_pairs = negative_pairs.sort_values('treat score', ascending = True)
        for prob in all_pos_test_probs:
            rank = len(negative_pairs) - bisect.bisect_left(list(negative_pairs['treat score']), prob) + 1
            # Add to total
            mrr_total += 1/rank
            for idx, k in enumerate(k_lst):
                if rank <= k:
                    hitk_total_lst[idx] += 1
    
    # Computing scores and curves
    y_true = df_all['y']
    y_score = df_all['treat score']
    
    # Compute AUROC, AP, MRR and Hit@k scores
    auroc = skl.metrics.roc_auc_score(y_true, y_score)
    ap = skl.metrics.average_precision_score(y_true, y_score)
    mrr = mrr_total/len(test_diseases)
    hitk_lst = list(hitk_total_lst/len(test_diseases))
    
    # Prepare output
    out = [auroc, ap, mrr] + hitk_lst
    if is_return_curves:
        roc_tup = skl.metrics.roc_curve(y_true, y_score)
        prc_tup = skl.metrics.precision_recall_curve(y_true, y_score)
        out = out, roc_tup, prc_tup

    return out 