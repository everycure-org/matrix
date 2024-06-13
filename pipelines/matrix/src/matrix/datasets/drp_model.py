"""Module containing representations for drug repurposing models."""
import pandas as pd
import numpy as np
import abc

from sklearn.base import BaseEstimator
from sklearn.impute._base import _BaseImputer

from typing import Any, Dict, List, Union, Tuple

from refit.v1.core.make_list_regexable import make_list_regexable

from matrix.datasets.graph import KnowledgeGraph
from matrix.pipelines.modelling.nodes import _add_embeddings


class DRPmodel(abc.ABC):
    """An abstract class representing a model which gives a "treat" score to drug-disease pairs.

    FUTURE: predict_class method
    """

    @abc.abstractmethod
    def give_prob_score(self, pairs: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Appends a "treat score" column to a drug-disease DataFrame.

        Args:
            pairs: Drug-disease pairs for which to generate scores.
                Column names are 'source' for drugs and 'target' for diseases.
            kwargs: Extra arguments .
        """
        ...


class DRPmodel3class(DRPmodel):
    """An abstract class representing a 3-class drug-purposing model.

    FUTURE: custom label names
    """

    @abc.abstractmethod
    def give_all_scores(self, pairs: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Appends columns corresponding to the three probability scores.

        Args:
            pairs: Drug-disease pairs for which to generate scores.
                Column names are 'source' for drugs and 'target' for diseases.
            kwargs: Extra arguments .
        """
        ...

    def give_prob_score(self, pairs: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """See DRPmodel.give_treat_scores docstring."""
        return self.give_all_scores(pairs, **kwargs).drop(
            labels=["not treat score", "unknown score"], axis=1
        )


class DRPmodel3classScikit(DRPmodel3class):
    """A concrete class representing a 3-class DRP model using an sklearn estimator.

    FUTURE: custom label names
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        graph: KnowledgeGraph,
        transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
        features: List[str],
    ) -> None:
        """Initialises a DRPmodel3classScikit instance.

        Args:
            estimator: sklearn estimator.
            graph: List of features, may be regex specified.
            transformers: Dictionary of fitted transformers.
            features: List of features, may be regex specified.
        """
        self.estimator = estimator
        self.graph = graph
        self.transformers = transformers
        self.features = features

    def _vectorise_transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Collects and transforms embeddings for drug-disease pairs.

        Args:
            pairs: Drug-disease pairs for which to generate scores.
                Column names are 'source' for drugs and 'target' for diseases.

        Returns:
            The drug-disease pairs DataFrame with extra columns representing the transformed embeddings.
        """
        pairs = pairs.copy()

        # Add embeddings
        pairs = _add_embeddings(pairs, graph)

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
    def _give_all_scores_arr(
        self,
        features: List[str],
        pairs_vect: pd.DataFrame,
        enable_regex: bool = True,
    ) -> np.array:
        """Computes a 2d array representing the probability scores for all drug-disease pairs.

        Args:
            features: List of features, may be regex specified.
            pairs_vect: Drug-disease pairs DataFrame containing transformed features.
            enable_regex: Enable regex for features.
        """
        all_scores = self.estimator.predict_proba(pairs_vect[features].values)
        return all_scores

    def give_all_scores(
        self,
        pairs: pd.DataFrame,
        skip_vectorise=False,
    ) -> pd.DataFrame:
        """Appends columns corresponding to the three probability scores.

        Args:
            pairs: Drug-disease pairs for which to generate scores.
                Column names are 'source' for drugs and 'target' for diseases.
            skip_vectorise (optional): Skip vectorisation step if pairs already contains transformed
                embedding columns. Defaults to False.
        """
        # Vectorise and compute scores
        if not skip_vectorise:
            pairs_vect = self._vectorise_transform(pairs)
        else:
            pairs_vect = pairs
        scores_arr = self._give_all_scores_arr(self.features, pairs_vect)

        # Append columns
        pairs = pairs.copy()
        pairs["not treat score"] = scores_arr[:, 0]
        pairs["treat score"] = scores_arr[:, 1]
        pairs["unknown score"] = scores_arr[:, 2]
        return pairs
