"""Module containing representations for drug repurposing models."""
import pandas as pd
import abc

from sklearn.base import BaseEstimator
from sklearn.impute._base import _BaseImputer

from typing import Any, Dict, List, Union, Tuple


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
        return give_all_scores(self, pairs).drop(labels = ['not treat score', 'unknown score'], axis = 1)
    

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
                 ):
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

    def give_all_scores(self, pairs : pd.DataFrame, skip_vectorise = False):
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

        TO DO: quick test
        """

        if not skip_vectorise:
            pairs_vect = pairs.copy()

            # Add embeddings
            pairs_vect["source_embedding"] = pairs_vect.apply(
                lambda row: graph._embeddings[row.source], axis=1
            )
            pairs_vect["target_embedding"] = pairs_vect.apply(
                lambda row: graph._embeddings[row.target], axis=1
            )

            for _, transform in transformers.items():
                # Apply transformer
                features_selected = pairs_vect[features]
                transformed = pd.DataFrame(
                    transformer.transform(features_selected),
                    index=features_selected.index,
                    columns=transformer.get_feature_names_out(features_selected),
                )

                # Overwrite columns
                pairs_vect = pd.concat(
                    [pairs_vect.drop(columns=features), transformed],
                    axis="columns",
                )

        # Compute scores and add columns
        all_scores = estimator.predict_proba(data_vect[features].values)
        data['not treat score'] = all_scores[:,0]
        data['treat score'] = all_scores[:,1]
        data['unknown score'] = all_scores[:,2]
        return data 

            


