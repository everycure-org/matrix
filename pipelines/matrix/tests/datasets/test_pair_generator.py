import numpy as np
import pandas as pd
import pytest
from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import (
    FullMatrixPositives,
    GroundTruthTestPairs,
    MatrixTestDiseases,
    RandomDrugDiseasePairGenerator,
    ReplacementDrugDiseasePairGenerator,
)
from matrix.pipelines.modelling.nodes import make_folds
from sklearn.model_selection import ShuffleSplit

## Test negative sampling pair generators


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
            ["drug:drug_1", "disease:disease_1", 1, [1, 0, 0, 0], [0, 0, 1, 0]],
            ["drug:drug_2", "disease:disease_2", 1, [0, 1, 0, 0], [0, 0, 0, 1]],
            ["drug:drug_3", "disease:disease_2", 1, [0, 2, 0, 0], [0, 0, 0, 2]],
        ],
        columns=["source", "target", "y", "source_embedding", "target_embedding"],
    )


def test_random_drug_disease_pair_generator(graph: KnowledgeGraph, known_pairs: pd.DataFrame):
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

    known_pairs_split = make_folds(
        known_pairs,
        splitter,
    )
    known_pairs_split = known_pairs_split[known_pairs_split["fold"] == 0]

    # When generating unknown pairs
    unknown = generator.generate(graph, known_pairs_split)

    # Then set of unknown pairs generated, where generated pairs
    # are distinct from known pairs and are always (drug, disease) pairs.
    known_positives_train = known_pairs_split[(known_pairs_split["y"] == 1) & (known_pairs_split["split"] == "TRAIN")]
    assert unknown.shape[0] == 2 * n_replacements * len(known_positives_train)
    assert pd.merge(known_pairs, unknown, how="inner", on=["source", "target"]).empty
    assert set(unknown["source"].to_list()).issubset(graph._drug_nodes)
    assert set(unknown["target"].to_list()).issubset(graph._disease_nodes)


## Test evaluation dataset generators


@pytest.fixture
def matrix():
    """Fixture to create a sample matrix for testing."""
    np.random.seed(42)  # For reproducibility

    # Create a sample matrix with 100 rows
    drugs = [f"drug_{i}" for i in range(10)]
    diseases = [f"disease_{i}" for i in range(10)]
    pairs = [(drug, disease) for drug in drugs for disease in diseases]
    data = {
        "source": [pair[0] for pair in pairs],
        "target": [pair[1] for pair in pairs],
        "treat_score": np.random.rand(100),
        "is_positive_1": np.random.choice([True, False], 100, p=[0.3, 0.7]),
        "is_positive_2": np.random.choice([True, False], 100, p=[0.3, 0.7]),
        "is_negative_1": np.random.choice([True, False], 100, p=[0.3, 0.7]),
        "is_negative_2": np.random.choice([True, False], 100, p=[0.3, 0.7]),
    }
    df = pd.DataFrame(data).sort_values("treat_score", ascending=False)

    # Ensure no overlap between positive and negative pairs
    all_cols = ["is_positive_1", "is_positive_2", "is_negative_1", "is_negative_2"]
    for col in all_cols:
        other_cols = [c for c in all_cols if c != col]
        df.loc[df[col], other_cols] = False

    return df


def test_ground_truth_test_pairs(matrix):
    # Given a matrix and a test data generator
    generator = GroundTruthTestPairs(
        positive_columns=["is_positive_1", "is_positive_2"],
        negative_columns=["is_negative_1", "is_negative_2"],
    )

    # When generating the test dataset
    generated_data = generator.generate(matrix)

    # Then generated test data contains all positive and negative pairs
    expected_positives = matrix[matrix["is_positive_1"] | matrix["is_positive_2"]]
    expected_negatives = matrix[matrix["is_negative_1"] | matrix["is_negative_2"]]
    assert len(generated_data) == len(expected_positives) + len(expected_negatives)
    assert generated_data[generated_data["y"] == 1].shape[0] == len(expected_positives)
    assert generated_data[generated_data["y"] == 0].shape[0] == len(expected_negatives)
    assert set(generated_data.columns) == set(matrix.columns).union({"y"})


def test_matrix_test_diseases(matrix):
    # Given a matrix and a test data generator
    generator = MatrixTestDiseases(
        positive_columns=["is_positive_1", "is_positive_2"],
        removal_columns=["is_negative_1"],
    )

    # When generating the test dataset
    generated_data = generator.generate(matrix)

    # Then generated test data:
    #   - contains all rows where the target is in the positive pairs set
    #   - does not include rows marked for removal
    #   - has the correct 'y' labels
    positive_pairs = matrix[matrix["is_positive_1"] | matrix["is_positive_2"]]
    positive_diseases = positive_pairs["target"].unique()
    expected_data = matrix[matrix["target"].isin(positive_diseases) | ~matrix["is_negative_1"]]
    expected_data["y"] = (expected_data["is_positive_1"] | expected_data["is_positive_2"]).astype(int)

    assert len(generated_data) == len(expected_data)
    assert set(generated_data["target"]).issubset(set(positive_diseases))
    assert (generated_data["y"] == expected_data["y"]).all()
    assert set(generated_data.columns) == set(matrix.columns).union({"y"})


def test_full_matrix_positives(matrix):
    # Given a matrix and a full matrix positives generator
    generator = FullMatrixPositives(
        positive_columns=["is_positive_1", "is_positive_2"],
        removal_columns=["is_negative_1"],
    )

    # When generating the dataset
    generated_data = generator.generate(matrix)
    # Then generated data:
    #   - contains only positive pairs
    #   - has correct 'y', 'rank', and 'non_pos_quantile_rank' columns
    #   - maintains the original order of the matrix
    expected_positives = matrix[matrix["is_positive_1"] | matrix["is_positive_2"]]
    expected_removed = matrix[matrix["is_negative_1"]]

    assert len(generated_data) == len(expected_positives)
    assert (generated_data["y"] == 1).all()
    assert list(generated_data["source"]) == list(expected_positives["source"])
    assert list(generated_data["target"]) == list(expected_positives["target"])
    assert {"y", "rank", "non_pos_rank", "non_pos_quantile_rank"}.issubset(set(generated_data.columns))
    assert generated_data["treat_score"].is_monotonic_decreasing
    assert (generated_data["non_pos_quantile_rank"].between(0, 1)).all()
    # Check that the non_pos_rank is computed against non-positive non-removed pairs
    assert (
        generated_data["non_pos_rank"]
        .between(1, len(matrix) - len(expected_removed) - len(expected_positives) + 1)
        .all()
    )
