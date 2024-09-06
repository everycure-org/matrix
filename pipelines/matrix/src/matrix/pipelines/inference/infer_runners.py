"""Utils for inference running."""
from abc import ABC, abstractmethod
from typing import List, Type
import pandas as pd


class inferRunner(ABC):
    """Base class for running inference."""

    def __init__(self):
        """Initiate base class."""
        pass

    @abstractmethod
    def ingest(
        self,
        model,
        nodes,
    ):
        """Ingest the model, nodes, drug list and disease list of interest."""
        pass

    @abstractmethod
    def run_inference(self):
        """Run inference."""
        pass

    @abstractmethod
    def cross_check(self, train_df):
        """Cross check with training data and flag/remove drug-disease pairs used for training."""
        df = self._scores
        if "training_data" in df.columns:
            raise ValueError(
                "cross-check already executed (i.e. detected training_data col in the scores)"
            )
        train_df = train_df.loc[train_df.split == "TRAIN"]
        train_pairs = set(zip(train_df["drug"], train_df["disease"]))
        df["training_data"] = df.apply(
            lambda row: (row["drug"], row["disease"]) in train_pairs, axis=1
        )
        self._scores = df

    @abstractmethod
    def generate_stats(
        self,
    ):
        """Generating descriptive statistics."""
        pass

    @abstractmethod
    def visualise_scores(self, scores):
        """Base class for visualising scores."""
        pass


class inferPerPair(inferRunner):
    """Class for running inference for a single indication."""

    def __init__(
        self,
        drug_id,
        disease_id,
    ):
        """Initiate base class."""
        self._drug = drug_id
        self._disease = disease_id

    def ingest_data(self, nodes):
        """Ingest the model, nodes, drug list and disease list of interest."""
        drug = nodes.loc[nodes.id == self._drug]
        disease = nodes.loc[nodes.id == self._disease]
        self._input = np.concatenate([drug, disease])

    def run_inference(self, model):
        """Run inference."""
        score = model(self.input)
        df = pd.DataFrame(
            {"drug": self._drug, "disease": self._disease, "treat_score": score}
        )
        self._scores = df


class inferPerDisease(inferRunner):
    """Class for running inference for a several indications (e.g. subtypes of the same disease)."""

    def __init__(
        self,
        drug_id: List,
        disease_id: LIst,
    ):
        """Initiate base class."""
        self._drug = drug_id
        self._disease = disease_id

    def ingest(
        self,
        model,
        nodes,
    ):
        """Ingest the model, nodes, drug list and disease list of interest."""
        pass

    def run_inference(self):
        """Run inference."""
        pass


# class inferPerSubtypes(inferRunner):
#     """Class for running inference for several indications per therapy area."""


# ### UTILS WIP

# def get_ranked_drugs(model, drug_ids_lst, disease_id, embedding_array, node_to_index):
#     """
#     Gives sorted list of "treat" probability scores for a collection of drugs and a single disease
#     """
#     pairs = pd.DataFrame({'source': drug_ids_lst,'target': disease_id})
#     pairs['probs'] = get_probabilities(pairs, model, embedding_array, node_to_index)[:,1]
#     pairs_sorted = pairs.sort_values('probs', ascending = False).reset_index(drop=True)
#     drugs_sorted = pairs_sorted.drop(columns='target')
#     return drugs_sorted

# def restrict_node_type(node_df, node_type_lst):
#     """
#     Returns KG nodes restricted to a list of categories.
#     """
#     node_df['clean_category'] = node_df['category'].str.contains(('|').join(node_type_lst))
#     return node_df[node_df['clean_category']==True]

# def get_ranked_drugs(model, drug_ids_lst, disease_id, embedding_array, node_to_index):
#     """
#     Gives sorted list of "treat" probability scores for a collection of drugs and a single disease
#     """
#     pairs = pd.DataFrame({'source': drug_ids_lst,'target': disease_id})
#     pairs['probs'] = get_probabilities(pairs, model, embedding_array, node_to_index)[:,1]
#     pairs_sorted = pairs.sort_values('probs', ascending = False).reset_index(drop=True)
#     drugs_sorted = pairs_sorted.drop(columns='target')
#     return drugs_sorted
