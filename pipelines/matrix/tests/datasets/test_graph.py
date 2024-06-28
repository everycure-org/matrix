import pytest

import pandas as pd
from sklearn.model_selection import ShuffleSplit

from matrix.datasets.graph import (
    KnowledgeGraph,
    RandomDrugDiseasePairGenerator,
    ReplacementDrugDiseasePairGenerator,
)
from matrix.pipelines.modelling.nodes import make_splits


@pytest.fixture(name="graph")
def graph_fixture() -> KnowledgeGraph:
    """Fixture to yield a dummy knowledge graph."""
    return KnowledgeGraph(
        pd.DataFrame(
            [
                ["drug:drug_1", True, False, [1, 0, 0, 0]],
                ["drug:drug_2", True, False, [0, 1, 0, 0]],
                ["disease:disease_1", False, True, [0, 0, 1, 0]],
                ["disease:disease_2", False, True, [0, 0, 0, 1]],
            ],
            columns=["id", "is_drug", "is_disease", "embedding"],
        )
    )


@pytest.fixture(name="known_pairs")
def known_pairs_fixture() -> pd.DataFrame:
    """Fixture to yield a dummy known pairs dataset."""
    return pd.DataFrame(
        [
            ["drug:drug_1", "disease:disease_1", 1],
            ["drug:drug_2", "disease:disease_2", 1],
        ],
        columns=["source", "target", "y"],
    )


def test_random_drug_disease_pair_generator(
    graph: KnowledgeGraph, known_pairs: pd.DataFrame
):
    # Given a random drug disease pair generator
    generator = RandomDrugDiseasePairGenerator(
        random_state=42,
        n_unknown=2,
        y_label=2,
        drug_flags=["is_drug"],
        disease_flags=["is_disease"],
    )

    # When generating unknown pairs
    unknown = generator.generate(graph, known_pairs)

    # Then set of unknown pairs generated, where generated pairs
    # are distinct from known pairs and are always (drug, disease) pairs.
    assert unknown.shape[0] == 2
    assert pd.merge(known_pairs, unknown, how="inner", on=["source", "target"]).empty
    assert set(unknown["source"].to_list()).issubset(graph._drug_nodes)
    assert set(unknown["target"].to_list()).issubset(graph._disease_nodes)


@pytest.mark.parametrize(
    "n_replacements,splitter",
    [(2, ShuffleSplit(n_splits=1, test_size=0.5, random_state=111))],
)
def test_replacement_drug_disease_pair_generator(
    graph: KnowledgeGraph, known_pairs: pd.DataFrame, n_replacements, splitter, spark
):
    # Given a replacement drug disease pair generator and a test-train split for the known data
    generator = ReplacementDrugDiseasePairGenerator(
        random_state=42,
        n_replacements=n_replacements,
        y_label=2,
        drug_flags=["is_drug"],
        disease_flags=["is_disease"],
    )
    known_pairs_split = make_splits(spark.createDataFrame(known_pairs), splitter)

    # When generating unknown pairs
    unknown = generator.generate(graph, known_pairs_split)

    # Then set of unknown pairs generated, where generated pairs
    # are distinct from known pairs and are always (drug, disease) pairs.
    known_positives_test = known_pairs_split[
        (known_pairs_split["y"] == 1) & (known_pairs_split["split"] == "TRAIN")
    ]
    assert unknown.shape[0] == 2 * n_replacements * len(known_positives_test)
    assert pd.merge(known_pairs, unknown, how="inner", on=["source", "target"]).empty
    assert set(unknown["source"].to_list()).issubset(graph._drug_nodes)
    assert set(unknown["target"].to_list()).issubset(graph._disease_nodes)
