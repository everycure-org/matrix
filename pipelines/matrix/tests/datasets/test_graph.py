import pytest

import pandas as pd

from matrix.datasets.graph import KnowledgeGraph, RandomDrugDiseasePairGenerator


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
            ["drug:drug_1", "disease:disease_1"],
            ["drug:drug_2", "disease:disease_2"],
        ],
        columns=["source", "target"],
    )


def test_random_drug_disease_pair_generator(graph, known_pairs):
    # Given a random drug disease pair generator
    generator = RandomDrugDiseasePairGenerator(random_state=42, n_unknown=2)

    # When generating unknown pairs
    unknown = generator.generate(graph, known_pairs)

    # Then set of unknown pairs generated, where generated pairs
    # are distinct from known pairs and are always (drug, disease) pairs.
    assert pd.merge(known_pairs, unknown, how="inner", on=["source", "target"]).empty
    assert set(unknown["source"].to_list()).issubset(graph._drug_nodes)
    assert set(unknown["target"].to_list()).issubset(graph._disease_nodes)
