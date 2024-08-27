import pytest

import pandas as pd
from sklearn.model_selection import ShuffleSplit

from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import (
    RandomDrugDiseasePairGenerator,
    ReplacementDrugDiseasePairGenerator,
    MatrixTestDiseases,
    GroundTruthTestPairs,
)
from matrix.pipelines.modelling.nodes import make_splits

from pyspark.sql.types import StructType


@pytest.fixture(name="graph")
def graph_fixture() -> KnowledgeGraph:
    """Fixture to yield a dummy knowledge graph."""
    return KnowledgeGraph(
        pd.DataFrame(
            [
                ["drug:drug_1", True, False, [1, 0, 0, 0]],
                ["drug:drug_2", True, False, [0, 1, 0, 0]],
                ["drug:drug_3", True, False, [0, 2, 0, 0]],
                ["disease:disease_1", False, True, [0, 0, 1, 0]],
                ["disease:disease_2", False, True, [0, 0, 0, 1]],
                ["disease:disease_3", False, True, [0, 0, 0, 2]],
            ],
            columns=["id", "is_drug", "is_disease", "topological_embedding"],
        )
    )


@pytest.fixture(name="known_pairs")
def known_pairs_fixture() -> pd.DataFrame:
    """Fixture to yield a dummy known pairs dataset."""
    return pd.DataFrame(
        [
            ["drug:drug_1", "disease:disease_1", 1],
            ["drug:drug_2", "disease:disease_2", 1],
            ["drug:drug_3", "disease:disease_2", 1],
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

    known_pairs_split = make_splits(
        graph,
        known_pairs,
        splitter,
    )

    # When generating unknown pairs
    unknown = generator.generate(graph, known_pairs_split)

    # Then set of unknown pairs generated, where generated pairs
    # are distinct from known pairs and are always (drug, disease) pairs.
    known_positives_train = known_pairs_split[
        (known_pairs_split["y"] == 1) & (known_pairs_split["split"] == "TRAIN")
    ]
    assert unknown.shape[0] == 2 * n_replacements * len(known_positives_train)
    assert pd.merge(known_pairs, unknown, how="inner", on=["source", "target"]).empty
    assert set(unknown["source"].to_list()).issubset(graph._drug_nodes)
    assert set(unknown["target"].to_list()).issubset(graph._disease_nodes)


@pytest.mark.parametrize(
    "splitter",
    [ShuffleSplit(n_splits=1, test_size=2 / 3, random_state=1)],
)
def test_ground_truth_test_pairs(
    graph: KnowledgeGraph, known_pairs: pd.DataFrame, splitter, spark
):
    # Given a test-train split for the known data and a test data generator
    generator = GroundTruthTestPairs()
    known_pairs_split = make_splits(
        graph,
        known_pairs,
        splitter,
    )

    # When generating the test dataset
    generated_data = generator.generate(graph, known_pairs_split)

    # Then generated test data is equal to the test set of the known pairs
    known_test = known_pairs_split[known_pairs_split["split"] == "TEST"]
    assert generated_data.shape == known_test.shape
    assert (generated_data["source"] == known_test["source"]).all()
    assert (generated_data["target"] == known_test["target"]).all()
    assert (generated_data["y"] == known_test["y"]).all()


@pytest.mark.parametrize(
    "splitter",
    [ShuffleSplit(n_splits=1, test_size=2 / 3, random_state=1)],
)
def test_matrix_test_diseases(
    graph: KnowledgeGraph, known_pairs: pd.DataFrame, splitter, spark
):
    # Given a list of drugs, a test-train split for the known data and a test data generator
    generator = MatrixTestDiseases(["is_drug"])
    known_pairs_split = make_splits(
        graph,
        known_pairs,
        splitter,
    )

    # When generating the test dataset
    generated_data = generator.generate(graph, known_pairs_split)

    # Then generated test data:
    #   - has the correct length,
    #   - does not contain test data,
    #   - the drug always lies in the given drug list,
    #   - the disease always appears in the known positive test set.
    #   - the number of data-points labeled with y=1 is equal to the number of known positive test pairs
    drugs_lst = graph._drug_nodes
    known_positives_test = known_pairs_split[
        (known_pairs_split["y"] == 1) & (known_pairs_split["split"] == "TEST")
    ]
    known_pos_test_diseases = list(known_positives_test["target"].unique())
    known_train = known_pairs_split[known_pairs_split["split"] == "TRAIN"]
    known_train_in_matrix = known_train[
        known_train["source"].isin(drugs_lst)
        & known_train["target"].isin(known_pos_test_diseases)
    ]
    assert len(generated_data) == len(drugs_lst) * len(known_pos_test_diseases) - len(
        known_train_in_matrix
    )
    assert pd.merge(
        generated_data, known_train, how="inner", on=["source", "target"]
    ).empty
    assert generated_data["source"].isin(drugs_lst).all()
    assert generated_data["target"].isin(known_pos_test_diseases).all()
    assert generated_data["y"].sum() == len(known_positives_test)
