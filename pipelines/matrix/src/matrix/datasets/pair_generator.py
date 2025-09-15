import abc
import random
from typing import List, Set, Tuple, Union

import pandas as pd
from tqdm import tqdm

from matrix.datasets.graph import KnowledgeGraph


class DrugDiseasePairGenerator(abc.ABC):
    """Generator strategy class to represent drug-disease pair generators."""

    @abc.abstractmethod
    def generate(self, known_pairs: pd.DataFrame) -> pd.DataFrame:
        """Function to generate drug-disease pairs from the knowledge graph.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with ground truth drug-disease pairs.

        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        ...


class SingleLabelPairGenerator(DrugDiseasePairGenerator):
    """Class representing generators outputting drug-disease pairs with a single label."""

    def __init__(self, y_label: int, random_state: int) -> None:
        """Initializes a SingleLabelPairGenerator instance.

        Args:
            y_label: label to assign to generated pairs.
            random_state: Random seed.
        """
        self._y_label = y_label
        self._random_state = random_state
        random.seed(random_state)


def _sample_random_pairs(
    graph: KnowledgeGraph,
    known_data_set: Set[tuple],
    drug_samp_ids: List[str],
    disease_samp_ids: List[str],
    n_unknown: int,
    y_label: int,
) -> pd.DataFrame:
    """Helper function to sample random drug-disease pairs.

    Args:
        graph: KnowledgeGraph instance.
        known_data_set: Set of known drug-disease pairs to avoid.
        drug_samp_ids: List of drug IDs to sample from.
        disease_samp_ids: List of disease IDs to sample from.
        n_unknown: Number of unknown pairs to generate.
        y_label: Label to assign to generated pairs.

    Returns:
        DataFrame with sampled drug-disease pairs.
    """
    unknown_data = []
    while len(unknown_data) < n_unknown:
        drug = random.choice(drug_samp_ids)
        disease = random.choice(disease_samp_ids)

        if (drug, disease) not in known_data_set:
            unknown_data.append(
                [
                    drug,
                    graph.get_embedding(drug)["rtxkg2"],
                    graph.get_embedding(drug)["robokop"],
                    disease,
                    graph.get_embedding(disease)["rtxkg2"],
                    graph.get_embedding(disease)["robokop"],
                    y_label,
                ]
            )

    return pd.DataFrame(
        columns=[
            "source",
            "source_rtxkg2_embedding",
            "source_robokop_embedding",
            "target",
            "target_rtxkg2_embedding",
            "target_robokop_embedding",
            "y",
        ],
        data=unknown_data,
    )


## Generators for negative sampling during training


class RandomDrugDiseasePairGenerator(SingleLabelPairGenerator):
    """Random drug-disease pair implementation.

    Strategy implementing a drug-disease pair generator using randomly sampled drugs and diseases.

    """

    def __init__(
        self,
        y_label: int,
        random_state: int,
        n_unknown: int,
        drug_flags: List[str],
        disease_flags: List[str],
    ) -> None:
        """Initializes the RandomDrugDiseasePairGenerator instance.

        Args:
            y_label: label to assign to generated pairs.
            random_state: Random seed.
            n_unknown: Number of unknown drug-disease pairs to generate.
            drug_flags: List of knowledge graph flags defining drugs sample set.
            disease_flags: List of knowledge graph flags defining diseases sample set.
        """
        self._n_unknown = n_unknown
        self._drug_flags = drug_flags
        self._disease_flags = disease_flags
        super().__init__(y_label, random_state)

    def generate(self, graph: KnowledgeGraph, known_pairs: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Function to generate drug-disease pairs according to the strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.
            kwargs: additional kwargs to use
        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        # Define ground truth dataset
        known_data_set = {(drug, disease) for drug, disease in zip(known_pairs["source"], known_pairs["target"])}

        # Defining list of node id's to sample from
        drug_samp_ids = graph.flags_to_ids(self._drug_flags)
        disease_samp_ids = graph.flags_to_ids(self._disease_flags)

        return _sample_random_pairs(
            graph, known_data_set, drug_samp_ids, disease_samp_ids, self._n_unknown, self._y_label
        )


class ReplacementDrugDiseasePairGenerator(SingleLabelPairGenerator):
    """Replacement drug-disease pair implementation.

    Strategy implementing a drug-disease pair generator using random drug and disease replacements.

    """

    def __init__(
        self,
        y_label: int,
        random_state: int,
        n_replacements: int,
        drug_flags: List[str],
        disease_flags: List[str],
    ) -> None:
        """Initializes the ReplacementDrugDiseasePairGenerator instance.

        Args:
            y_label: label to assign to generated pairs.
            random_state: Random seed.
            n_replacements: Number of replacements to make.
            drug_flags: List of knowledge graph flags defining drugs sample set.
            disease_flags: List of knowledge graph flags defining diseases sample set.
        """
        self._n_replacements = n_replacements
        self._drug_flags = drug_flags
        self._disease_flags = disease_flags
        super().__init__(y_label, random_state)

    def generate(self, graph: KnowledgeGraph, known_pairs: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Function to generate drug-disease pairs according to the strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.
            kwargs: additional kwargs to use
        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        known_data_set = {(drug, disease) for drug, disease in zip(known_pairs["source"], known_pairs["target"])}

        # Extract known positive training set
        kp_train_pairs = known_pairs[(known_pairs["y"] == 1) & (known_pairs["split"] == "TRAIN")]
        kp_train_set = {(drug, disease) for drug, disease in zip(kp_train_pairs["source"], kp_train_pairs["target"])}

        # Defining list of node id's to sample from
        drug_samp_ids = graph.flags_to_ids(self._drug_flags)
        disease_samp_ids = graph.flags_to_ids(self._disease_flags)

        # Generate unknown data
        unknown_data = []
        for kp_drug, kp_disease in tqdm(kp_train_set):
            unknown_data += ReplacementDrugDiseasePairGenerator._make_replacements(
                graph,
                kp_drug,
                kp_disease,
                drug_samp_ids,
                disease_samp_ids,
                self._n_replacements,
                known_data_set,
                self._y_label,
            )

        return pd.DataFrame(
            columns=[
                "source",
                "source_rtxkg2_embedding",
                "source_robokop_embedding",
                "target",
                "target_rtxkg2_embedding",
                "target_robokop_embedding",
                "y",
            ],
            data=unknown_data,
        )

    @staticmethod
    def _make_replacements(
        graph: KnowledgeGraph,
        kp_drug: str,
        kp_disease: str,
        drug_samp_ids: List[str],
        disease_samp_ids: List[str],
        n_replacements: int,
        known_data_set: Set[tuple],
        y_label: int,
    ) -> List[str]:
        """Helper function to generate list of drug-disease pairs through replacements."""
        # Sample pairs
        unknown_data = []
        while len(unknown_data) < 2 * n_replacements:
            # Sample a random drug and disease
            rand_drug = random.choice(drug_samp_ids)
            rand_disease = random.choice(disease_samp_ids)
            # Perform replacements
            if (kp_drug, rand_disease) not in known_data_set and (
                rand_drug,
                kp_disease,
            ) not in known_data_set:
                for drug, disease in [
                    (kp_drug, rand_disease),
                    (rand_drug, kp_disease),
                ]:
                    unknown_data.append(
                        [
                            drug,
                            graph.get_embedding(drug)["rtxkg2"],
                            graph.get_embedding(drug)["robokop"],
                            disease,
                            graph.get_embedding(disease)["rtxkg2"],
                            graph.get_embedding(disease)["robokop"],
                            y_label,
                        ]
                    )
        return unknown_data


class DiseaseSplitDrugDiseasePairGenerator(SingleLabelPairGenerator):
    """A pair generator that ensures negative sampling respects disease area splits.

    This generator ensures that diseases in the test set are never used for negative sampling
    during training, maintaining the integrity of disease area splits. This is important for
    proper evaluation of model performance on unseen disease areas.
    """

    def __init__(
        self,
        y_label: int,
        random_state: int,
        n_unknown: int,
        drug_flags: List[str],
        disease_flags: List[str],
    ) -> None:
        """Initializes the DiseaseSplitDrugDiseasePairGenerator instance.

        Args:
            y_label: label to assign to generated pairs.
            random_state: Random seed.
            n_unknown: Number of unknown drug-disease pairs to generate.
            drug_flags: List of knowledge graph flags defining drugs sample set.
            disease_flags: List of knowledge graph flags defining diseases sample set.
        """
        self._n_unknown = n_unknown
        self._drug_flags = drug_flags
        self._disease_flags = disease_flags
        super().__init__(y_label, random_state)

    def generate(self, graph: KnowledgeGraph, known_pairs: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Function to generate drug-disease pairs according to the strategy.

        Args:
            graph: KnowledgeGraph instance.
            known_pairs: DataFrame with known drug-disease pairs.
            kwargs: additional kwargs to use
        Returns:
            DataFrame with unknown drug-disease pairs.
        """
        # Define ground truth dataset
        known_data_set = {(drug, disease) for drug, disease in zip(known_pairs["source"], known_pairs["target"])}

        # Get training and test diseases for this fold
        train_diseases = set(known_pairs[known_pairs["split"] == "TRAIN"]["target"])
        test_diseases = set(known_pairs[known_pairs["split"] == "TEST"]["target"])

        # Get all diseases from graph that match disease flags
        all_diseases = set(graph.flags_to_ids(self._disease_flags))

        # Get training diseases that are in the graph and not in test set
        disease_samp_ids = [d for d in train_diseases if d in all_diseases and d not in test_diseases]

        if not disease_samp_ids:
            raise ValueError("No training diseases found in the knowledge graph")

        # Get drugs from graph that match drug flags
        drug_samp_ids = graph.flags_to_ids(self._drug_flags)

        # Sample pairs using the helper function
        unknown_df = _sample_random_pairs(
            graph, known_data_set, drug_samp_ids, disease_samp_ids, self._n_unknown, self._y_label
        )

        # Verify no test diseases appear in negative samples
        negative_diseases = set(unknown_df[unknown_df["y"] == self._y_label]["target"])
        test_diseases_in_negatives = test_diseases.intersection(negative_diseases)
        if test_diseases_in_negatives:
            raise ValueError(
                f"Test diseases found in negative samples: {test_diseases_in_negatives}. "
                "This indicates a potential data leakage issue."
            )

        return unknown_df


## Generators for evaluation datasets


class GroundTruthTestPairs(DrugDiseasePairGenerator):
    """Class representing ground truth test data."""

    def __init__(self, positive_columns: List[str], negative_columns: List[str]) -> None:
        """Initialises an instance of the class.

        Args:
            positive_columns: Names of the flag columns in the matrix which represent the positive pairs.
            negative_columns: Names of the flag columns in the matrix which represent the negative pairs.
        """
        self.positive_columns = positive_columns
        self.negative_columns = negative_columns

    def generate(
        self,
        matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Function to generate the dataset given a full matrix dataframe.

        Args:
            matrix: Pairs dataframe representing the full matrix with treat scores.

        Returns:
            Labelled ground truth drug-disease pairs dataset.
        """
        # Extract and label positive data
        positive_data_lst = []
        for col_name in self.positive_columns:
            positive_data_lst.append(matrix[matrix[col_name]].assign(y=1))

        # Extract and label negative data
        negative_data_lst = []
        for col_name in self.negative_columns:
            negative_data_lst.append(matrix[matrix[col_name]].assign(y=0))

        # Combine data
        data = pd.concat(positive_data_lst + negative_data_lst, ignore_index=True)

        # Return selected pairs
        return data


class MatrixTestDiseases(DrugDiseasePairGenerator):
    """A class representing dataset of pairs required for disease-specific ranking."""

    def __init__(
        self,
        positive_columns: List[str],
        removal_columns: Union[List[str], None] = None,
    ) -> None:
        """Initialises an instance of the class.

        Args:
            positive_columns: Names of the flag columns in the matrix which represent the positive pairs.
            removal_columns: Names of the flag columns in the matrix which represent the pairs to remove.
        """
        self.positive_columns = positive_columns
        self.removal_columns = removal_columns

    def generate(
        self,
        matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate dataset given a full matrix.

        Args:
            matrix: Pairs dataframe representing the full matrix with treat scores.

        Returns:
            Labelled drug-disease pairs dataset.
        """
        # Extract positive pairs and label in matrix
        is_positive = pd.Series(False, index=matrix.index)
        for col_name in self.positive_columns:
            is_positive = is_positive | matrix[col_name]
        positive_pairs = matrix[is_positive]
        matrix["y"] = is_positive.astype(int)

        # Restrict to diseases in the positive pairs set
        positive_diseases = positive_pairs["target"].unique()
        in_output = matrix["target"].isin(positive_diseases)

        # Remove flagged pairs
        if self.removal_columns is not None:
            is_remove = pd.Series(False, index=matrix.index)
            for col_name in self.removal_columns:
                is_remove = is_remove | matrix[col_name]
            in_output = in_output & ~is_remove

        # Apply boolean condition to matrix and return
        return matrix[in_output]


class FullMatrixPositives(DrugDiseasePairGenerator):
    """A class that represents the ranks of a set of positive pairs in a full matrix.

    FUTURE: We may want to add an option to remove selected pairs (e.g. other known positives).
    """

    def __init__(
        self,
        positive_columns: List[str],
        removal_columns: Union[List[str], None] = None,
    ) -> None:
        """Initialises an instance of the class.

        Args:
            positive_columns: Names of the flag columns in the matrix which represent the positive pairs.
            removal_columns: Names of the flag columns in the matrix which represent the pairs to remove.
        """
        self.positive_columns = positive_columns
        self.removal_columns = removal_columns

    def generate(
        self,
        matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate dataset given a full matrix.

        Args:
            matrix: Pairs dataframe representing the full matrix with treat scores.

        Returns:
            Labelled drug-disease pairs dataset.
        """
        # Remove flagged pairs; ensure index is reset
        matrix = matrix.reset_index(drop=True)
        if self.removal_columns is not None:
            is_remove = pd.Series(False, index=matrix.index)
            for col_name in self.removal_columns:
                is_remove = is_remove | matrix[col_name]
            matrix = matrix[~is_remove].reset_index(drop=True)

        # Extract and label positive pairs
        is_positive = pd.Series(False, index=matrix.index)
        for col_name in self.positive_columns:
            is_positive = is_positive | matrix[col_name]
        positive_pairs = matrix[is_positive].assign(y=1)

        # Remove contribution from known positives to compute the rank against non-positive pairs
        positive_pairs["rank"] = positive_pairs.index + 1
        positive_pairs = positive_pairs.reset_index(drop=True)
        positive_pairs["non_pos_rank"] = positive_pairs["rank"] - positive_pairs.index

        # Compute the quantile rank against non-positive pairs
        num_non_pos = len(matrix[~is_positive])
        positive_pairs["non_pos_quantile_rank"] = (positive_pairs["non_pos_rank"] - 1) / num_non_pos
        return positive_pairs


class OnlyOverlappingPairs(DrugDiseasePairGenerator):
    """Class to generate pairs that overlap across all matrices for top n.

    Required for spearman rank, hypergeometric test and rank-commonality metrics."""

    def __init__(self, top_n: int = 1000) -> None:
        """Initialises an instance of the class.

        Args:
            top_n: Number of top pairs to be used for stability comparison.
        """
        self.top_n = top_n

    def _modify_matrices(self, matrices: Tuple[pd.DataFrame], score_col_name: str) -> List[pd.DataFrame]:
        """Modify matrices to create id column and sort by treat score.

        Args:
            matrices: DataFrames to be used for stability comparison.
        Returns:
            List of modified matrices.
        """
        new_matrices = []
        for matrix in matrices:
            matrix = matrix.sort_values(by=score_col_name, ascending=False).head(self.top_n)
            matrix["pair_id"] = matrix["source"] + "|" + matrix["target"]
            new_matrices.append(matrix)
        return new_matrices

    def _get_overlapping_pairs(self, matrices: Tuple[pd.DataFrame]) -> Set:
        """Get pairs that overlap across all matrices for top n.

        Args:
            matrices: DataFrames to be used for stability comparison.
        Returns:
            Set containing overlapping ids from all matrices.
        """
        overlapping_ids = set(matrices[0]["pair_id"])
        for matrix in matrices[1:]:
            overlapping_ids.intersection_update(set(matrix["pair_id"]))
        return overlapping_ids

    def generate(self, matrices, score_col_name: str) -> List[pd.DataFrame]:
        """Generates a dataframes of pairs that overlap across all matrices for top n."""
        matrices = self._modify_matrices(matrices, score_col_name)
        overlapping_pairs = self._get_overlapping_pairs(matrices)
        return pd.DataFrame(overlapping_pairs, columns=["pair_id"])


class NoGenerator(DrugDiseasePairGenerator):
    """Class to generate no pairs.

    Dummy class as for commonality@k calculation we want our matrix output for top n to remain unchanged."""

    def __init__(self, top_n: int = 1000) -> None:
        """Initialises an instance of the class.

        Args:
            top_n: Number of top pairs to be used for stability comparison.
        """
        self.top_n = top_n

    def generate(self, matrices, score_col_name: str = None) -> List[pd.DataFrame]:
        """Generates an empty dataframe as we are not using a list of common pairs for commonality@k"""
        return pd.DataFrame({}, columns=["pair_id"])
