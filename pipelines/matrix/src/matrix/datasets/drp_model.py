"""Module containing representations for drug repurposing models."""
import pandas as pd
import numpy as np
import abc

from sklearn.base import BaseEstimator
from sklearn.impute._base import _BaseImputer

from typing import Any, Dict, List, Union, Tuple

from refit.v1.core.make_list_regexable import make_list_regexable

from matrix.datasets.graph import KnowledgeGraph


class DRPmodel(abc.ABC):
    """
    An abstract class representing a model which gives a "treat" score to drug-disease pairs. 
    FUTURE: Adding training_set attribute could make sense
    """
    @abc.abstractmethod
    def give_treat_scores(self, pairs : pd.DataFrame) -> pd.DataFrame: 
        """
        Appends a "treat score" column to a drug-disease DataFrame. 

        Args:
            pairs (pd.DataFrame): Drug-disease pairs for which to generate scores. 
                Column names are 'source' for drugs and 'target' for diseases.
        """
        ...


class DRPmodel3class(DRPmodel):
    """
    An abstract class representing a 3-class drug-purposing model. 
    """
    @abc.abstractmethod
    def give_all_scores(self, pairs : pd.DataFrame) -> pd.DataFrame: 
        """
        Appends three columns to a drug-disease DataFrame: 
            - "not treat score"
            - "treat score"
            - "unknown score"

        Args:
            pairs (pd.DataFrame): Drug-disease pairs for which to generate scores. 
                Column names are 'source' for drugs and 'target' for diseases.
        """
        ...

    def give_treat_scores(self, pairs : pd.DataFrame) -> pd.DataFrame: 
        """
        See DRPmodel.give_treat_scores docstring. 
        """
        return self.give_all_scores(pairs).drop(labels = ['not treat score', 'unknown score'], axis = 1)
    

class DRPmodel3classScikit(DRPmodel3class):
    """
    A class representing 3-class drug repurposing models given by estimators from the scikit-learn
    interface (e.g. instances of xgboost.XGBClassifier or sklearn.ensemble.RandomForestClassifier).  
    """
    def __init__(self, 
                 estimator : BaseEstimator,
                 graph: KnowledgeGraph, 
                 transformers : Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
                 features : List[str]
                 ) -> None: 
        """Initialises a DRPmodel3classScikit instance 

        Args:
            estimator (BaseEstimator): sklearn estimator.
            graph (KnowledgeGraph): List of features, may be regex specified.
            transformers (Dict[str, Dict[str, Union[_BaseImputer, List[str]]]]):
                Dictionary of fitted transformers.
            features (List[str]): List of features, may be regex specified.
        """
        self.estimator = estimator
        self.graph = graph
        self.transformers = transformers
        self.features = features

    def _vectorise_transform(self, pairs : pd.DataFrame) -> pd.DataFrame: 
        """Collects embeddings for drug and diseases in the pairs DataFrame, 
        applies transformations and appends columns. 

        Args:
            pairs (pd.DataFrame): Drug-disease pairs for which to generate scores. 
                Column names are 'source' for drugs and 'target' for diseases.
        """
        pairs = pairs.copy()

        # Add embeddings
        pairs["source_embedding"] = pairs.apply(
            lambda row: self.graph._embeddings[row.source], axis=1
        )
        pairs["target_embedding"] = pairs.apply(
            lambda row: self.graph._embeddings[row.target], axis=1
        )

        # Iterate over transformers
        for _, transform in self.transformers.items():
            features = transform["features"]

            # Apply transformer
            transformer = transform["transformer"]
            features_selected = pairs[features]
            transformed = pd.DataFrame(
                transformer.transform(features_selected),
                index=features_selected.index,
                columns=transformer.get_feature_names_out(features_selected),
            )

            # Overwrite columns
            pairs = pd.concat(
                [pairs.drop(columns=features), transformed],
                axis="columns",
            )

        return pairs
    
    @make_list_regexable(source_df="pairs_vect", make_regexable="features")
    def _give_all_scores_arr(self,
                               features : List[str], 
                               pairs_vect : pd.DataFrame,
                               enable_regex: bool = True,
                               ) -> np.array:
        """
        Helper method which computes 2d array with three columns representing the: 
            - "not treat score"
            - "treat score"
            - "unknown score"
        for all drug-disease pairs. 
            
        Args:
            features (List[str]): List of features, may be regex specified.
            pairs (pd.DataFrame): Drug-disease pairs DataFrame containing transformed features. 
            enable_regex (bool): Enable regex for features.
        """
        all_scores = self.estimator.predict_proba(pairs_vect[features].values)
        return all_scores
    
    def give_all_scores(self,
                        pairs : pd.DataFrame, 
                        skip_vectorise = False,
                        ) -> pd.DataFrame:
        """
        Appends three columns to a drug-disease DataFrame: 
            - "not treat score"
            - "treat score"
            - "unknown score"
            
        Args:
            pairs (pd.DataFrame): Drug-disease pairs for which to generate scores. 
                Column names are 'source' for drugs and 'target' for diseases.
            skip_vectorise (bool, optional): Skip vectorisation step if pairs already contains transformed 
                embedding columns. Defaults to False.
        """
        # Vectorise and compute scores
        if not skip_vectorise:
            pairs_vect =  self._vectorise_transform(pairs)
        else:
            pairs_vect = pairs
        scores_arr = self._give_all_scores_arr(self.features, pairs_vect) 

        # Append columns
        pairs['not treat score'] = scores_arr[:,0]
        pairs['treat score'] = scores_arr[:,1]
        pairs['unknown score'] = scores_arr[:,2]
        return pairs
    


            


