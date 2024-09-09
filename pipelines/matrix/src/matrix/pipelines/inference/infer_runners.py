"""Utils for inference running."""
from abc import ABC, abstractmethod
from typing import List, Type
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ..modelling.model import ModelWrapper

##########################
####### BASE MODEL #######
##########################


class inferRunner(ABC):
    """Base class for running inference."""

    def __init__(self):
        """Initiate base class."""
        pass

    @abstractmethod
    def ingest_data(self, nodes):
        """Ingest the model, nodes, drug list and disease list of interest."""
        pass

    @abstractmethod
    def run_inference(self):
        """Run inference."""
        pass

    @abstractmethod
    def add_metadata(self, train_df):
        """Cross check with training data and flag/remove drug-disease pairs used for training."""
        df = self._scores
        if "training_data" in df.columns:
            raise ValueError(
                "cross-check already executed (i.e. detected training_data col in the scores)"
            )
        train_df = train_df.loc[train_df.split == "TRAIN"]
        train_pairs = set(zip(train_df["source"], train_df["target"]))
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


##########################
#### SPECIFIC MODELS #####
##########################


class inferPerPair(inferRunner):
    """Class for running inference for a single indication."""

    # TODO: need to finish
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
        drug = nodes.loc[nodes.id.isin(self._drug.id)]
        disease = nodes.loc[nodes.id.isin(self._disease.id)]
        vectorized_input = [
            np.concatenate(
                [
                    disease.topological_embedding.values[0],
                    disease.topological_embedding.values[0],
                ]
            )
        ]

        # Ingest metadata for later
        self._input = vectorized_input
        self._drug = drug.id.values
        self._disease = disease.id.values
        self._disease_meta = disease.description.values
        self._drug_meta = drug.description.values
        self._disease_name = disease.name.values
        self._drug_name = drug.name.values

    def run_inference(self, model):
        """Run inference."""
        scores = model.predict_proba(np.array(self._input))[:, 1]
        print(scores)
        df = pd.DataFrame(
            {"drug": self._drug, "disease": self._disease, "treat_score": scores}
        )
        print(scores)
        self._scores = df


class inferPerDisease(inferRunner):
    """Class for running inference for a several indications (e.g. subtypes of the same disease)."""

    def __init__(
        self,
        drug_id: List = None,
        disease_id: List = None,
    ):
        """Initiate base class."""
        self._drug = drug_id
        self._disease = disease_id

    def ingest_data(self, nodes):
        """Ingest the model, nodes, drug list and disease list of interest."""
        drug = nodes.loc[nodes.id.isin(self._drug.id)]  # .topological_embedding.values
        disease = nodes.loc[
            nodes.id.isin(self._disease.id)
        ]  # .topological_embedding.values[0]
        vectorized_input = []
        for embed in drug.topological_embedding.values:
            vectorized_input.append(
                np.concatenate([embed, disease.topological_embedding.values[0]])
            )

        # Ingest metadata for later
        self._input = vectorized_input
        self._drug = drug.id.values
        self._disease = [disease.id.values[0] for _ in drug.id.values]
        self._disease_meta = [disease.description.values[0] for _ in drug.id.values]
        self._drug_meta = drug.description.values
        self._disease_name = [disease.name.values[0] for _ in drug.id.values]
        self._drug_name = drug.name.values

    def run_inference(self, model):
        """Run inference."""
        scores = model.predict_proba(np.array(self._input))[:, 1]
        print(scores)
        df = pd.DataFrame(
            {"drug": self._drug, "disease": self._disease, "treat_score": scores}
        )
        print(scores)
        self._scores = df

    def add_metadata(self, train_df):
        """Cross check with training data and flag/remove drug-disease pairs used for training."""
        df = self._scores
        if "training_data" in df.columns:
            raise ValueError(
                "cross-check already executed (i.e. detected training_data col in the scores)"
            )
        train_df = train_df.loc[train_df.split == "TRAIN"]

        # Cross check if the drug-disease pair is present in the training data
        train_pairs = set(zip(train_df["source"], train_df["target"]))
        df["training_data"] = df.apply(
            lambda row: (row["drug"], row["disease"]) in train_pairs, axis=1
        )

        # Add names
        print(len(self._drug_meta))
        print(len(self._disease_meta))
        print(len(self._drug_name))
        print(len(self._disease_name))
        df["drug_name"] = self._drug_name
        df["disease_name"] = self._disease_name

        # Add metadata (eg drug/disease description)
        df["drug_description"] = self._drug_meta
        df["disease_description"] = self._disease_meta

        self._scores = df  # loc[:, ['drug','drug_name','drug_description','disease','disease_name','disease_description','treat_score','training_data']]

    def generate_stats(
        self,
    ):
        """Generating descriptive statistics."""
        pass

    def visualise_scores(self, scores):
        """Base class for visualising scores."""
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.kdeplot(scores["treat_score"])
        ax.set_title("Distribution of Treatment Scores", fontsize=20, fontweight="bold")
        ax.set_xlabel("Treatment Score", fontsize=16)
        ax.set_ylabel("Frequency", fontsize=16)

        # Add gridlines for better readability
        ax.grid(True, linestyle="--", alpha=0.7)
        caption = (
            f"Mean: {np.mean(scores.treat_score)}, Std: {np.std(scores.treat_score)}, "
            f"Min: {min(scores.treat_score)}, Max: {max(scores.treat_score)}"
        )

        plt.figtext(0.5, 0.01, caption, ha="center", fontsize=14, fontstyle="italic")

        return fig


# TODO: implement drug centric class
