"""Module with nodes for evaluation."""
from tqdm import tqdm
from typing import List, Dict, Union

from sklearn.impute._base import _BaseImputer

import pandas as pd

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.make_list_regexable import _extract_elements_in_list

from matrix.datasets.graph import KnowledgeGraph

from matrix.pipelines.modelling.nodes import apply_transformers
from matrix.pipelines.evaluation.evaluation import Evaluation
from matrix.pipelines.modelling.model import ModelWrapper


@has_schema(
    schema={
        "source": "object",
        "target": "object",
    },
    allow_subset=True,
)
@inject_object()
def generate_pairs(
    graph: KnowledgeGraph,
    known_pairs: pd.DataFrame,
    drugs_lst_flags: List[str],
    diseases_lst_flags: List[str],
) -> pd.DataFrame:
    """Function to generate matrix dataset.

    Args:
        graph: KnowledgeGraph instance
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        drugs_lst_flags: List of flags defining the list of drugs.
        diseases_lst_flags: List of flags defining the list of drugs
    Returns:
        Pairs dataframe containing all combinations of drugs and diseases that do not lie in the training set.
    """
    # Collect list of drugs and diseases
    drugs_lst = graph.flags_to_ids(drugs_lst_flags)
    diseases_lst = graph.flags_to_ids(diseases_lst_flags)

    # Generate all combinations
    matrix_slices = []
    for disease in tqdm(diseases_lst):
        matrix_slice = pd.DataFrame({"source": drugs_lst, "target": disease})
        matrix_slices.append(matrix_slice)

    # Concatenate all slices at once
    matrix = pd.concat(matrix_slices, ignore_index=True)

    # Remove training set and return
    train_pairs = known_pairs[~known_pairs["split"].eq("TEST")]
    train_pairs_set = set(zip(train_pairs["source"], train_pairs["target"]))
    is_in_train = matrix.apply(
        lambda row: (row["source"], row["target"]) in train_pairs_set, axis=1
    )
    return matrix[~is_in_train]


def make_batch_predictions(
    graph: KnowledgeGraph,
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    score_col_name: str,
    batch_by: str = "target",
) -> pd.DataFrame:
    """Generate probability scores for drug-disease dataset.

    This function computes the scores in batches to avoid memory issues.

    Args:
        graph: Knowledge graph.
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        score_col_name: Probability score column name.
        batch_by: Column to use for batching (e.g., "target" or "source").

    Returns:
        Pairs dataset with additional column containing the probability scores.
    """

    def process_batch(batch: pd.DataFrame) -> pd.DataFrame:
        # Collect embedding vectors
        batch["source_embedding"] = batch.apply(
            lambda row: graph._embeddings[row.source], axis=1
        )
        batch["target_embedding"] = batch.apply(
            lambda row: graph._embeddings[row.target], axis=1
        )

        # Apply transformers to data
        transformed = apply_transformers(batch, transformers)

        # Extract features
        batch_features = _extract_elements_in_list(
            transformed.columns, features, raise_exc=True
        )

        # Generate model probability scores
        batch[score_col_name] = model.predict_proba(transformed[batch_features].values)[
            :, 1
        ]

        return batch[[score_col_name]]

    # Group data by the specified prefix
    grouped = data.groupby(batch_by)

    # Process data in batches
    result_parts = []
    for _, batch in tqdm(grouped):
        result_parts.append(process_batch(batch))

    # Combine results
    results = pd.concat(result_parts, axis=0)

    # Add scores to the original dataframe
    data[score_col_name] = results[score_col_name]

    return data


def make_predictions_and_sort(
    graph: KnowledgeGraph,
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    score_col_name: str,
    batch_by: str,
) -> pd.DataFrame:
    """Generate and sort probability scores for a drug-disease dataset.

    FUTURE: Perform parallelised computation instead of batching with a for loop.

    Args:
        graph: Knowledge graph.
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        score_col_name: Probability score column name.
        batch_by: Column to use for batching (e.g., "target" or "source").

    Returns:
        Pairs dataset sorted by an additional column containing the probability scores.
    """
    from sklearn.ensemble._forest import RandomForestClassifier

    # Generate scores
    data = make_batch_predictions(
        graph, data, transformers, model, features, score_col_name, batch_by=batch_by
    )

    # Sort by the probability score
    sorted_data = data.sort_values(by=score_col_name, ascending=False)
    return sorted_data


def generate_report(
    graph: KnowledgeGraph, data: pd.DataFrame, n_reporting: int
) -> pd.DataFrame:
    """Generates a report with the top pairs.

    Args:
        graph: Knowledge graph.
        data: Pairs dataset.
        n_reporting: Number of pairs in the report

    Returns:
        Dataframe containing the top pairs with additional information for the drugs and diseases.
    """
    # Select the top n_reporting rows
    top_pairs = data.head(n_reporting)

    # Add additional information for drugs and diseases (TODO: optimise for speed by e.g. caching or using Polars)
    top_pairs["drug_name"] = top_pairs["source"].apply(
        lambda x: graph.get_node_attribute(x, "name")
    )
    top_pairs["disease_name"] = top_pairs["target"].apply(
        lambda x: graph.get_node_attribute(x, "name")
    )
    top_pairs["drug_description"] = top_pairs["source"].apply(
        lambda x: graph.get_node_attribute(x, "des")
    )
    top_pairs["disease_description"] = top_pairs["target"].apply(
        lambda x: graph.get_node_attribute(x, "des")
    )

    # Rename ID columns
    top_pairs = top_pairs.rename(columns={"source": "drug_id"})
    top_pairs = top_pairs.rename(columns={"target": "disease_id"})

    # Reorder columns for better readability
    columns_order = [
        "drug_id",
        "drug_name",
        "drug_description",
        "disease_id",
        "disease_name",
        "disease_description",
    ] + [
        col
        for col in top_pairs.columns
        if col
        not in [
            "disease_id",
            "drug_name",
            "drug_description",
            "target",
            "disease_name",
            "disease_description",
        ]
    ]
    top_pairs = top_pairs[columns_order]

    return top_pairs
