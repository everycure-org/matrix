import numpy as np
import pandas as pd
import pytest
from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import (
    DiseaseSplitDrugDiseasePairGenerator,
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
    graph: KnowledgeGraph, known_pairs: pd.DataFrame, n_replacements, splitter
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


def test_disease_split_drug_disease_pair_generator(graph: KnowledgeGraph, known_pairs: pd.DataFrame):
    """Test that DiseaseSplitDrugDiseasePairGenerator only uses training diseases for negative sampling."""
    # Given a train-test split and the generator
    splitter = ShuffleSplit(n_splits=1, test_size=0.5, random_state=111)
    known_pairs_split = make_folds(known_pairs, splitter)
    known_pairs_split = known_pairs_split[known_pairs_split["fold"] == 0]

    generator = DiseaseSplitDrugDiseasePairGenerator(
        random_state=42,
        n_unknown=10,
        y_label=2,
        drug_flags=["is_drug"],
        disease_flags=["is_disease"],
    )

    # When generating unknown pairs
    unknown = generator.generate(graph, known_pairs_split)

    # Then:
    # 1. All generated pairs use diseases from the training set only
    train_diseases = set(known_pairs_split[known_pairs_split["split"] == "TRAIN"]["target"])
    assert set(unknown["target"]).issubset(train_diseases)

    # 2. No generated pairs exist in the known pairs set
    assert pd.merge(known_pairs, unknown, how="inner", on=["source", "target"]).empty

    # 3. All generated pairs are valid drug-disease pairs
    assert set(unknown["source"].to_list()).issubset(graph._drug_nodes)
    assert set(unknown["target"].to_list()).issubset(graph._disease_nodes)


@pytest.mark.parametrize(
    "test_size,expected_train_diseases",
    [
        (0.33, 2),  # One third of diseases in test set
        (0.5, 1),  # Half of diseases in test set
        (0.66, 1),  # Two thirds of diseases in test set
    ],
)
def test_disease_split_drug_disease_pair_generator_with_different_splits(
    graph: KnowledgeGraph, known_pairs: pd.DataFrame, test_size: float, expected_train_diseases: int
):
    """Test DiseaseSplitDrugDiseasePairGenerator with different train-test splits."""
    # Given a train-test split and the generator
    splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=111)
    known_pairs_split = make_folds(known_pairs, splitter)
    known_pairs_split = known_pairs_split[known_pairs_split["fold"] == 0]

    generator = DiseaseSplitDrugDiseasePairGenerator(
        random_state=42,
        n_unknown=10,
        y_label=2,
        drug_flags=["is_drug"],
        disease_flags=["is_disease"],
    )

    # When generating unknown pairs
    unknown = generator.generate(graph, known_pairs_split)

    # Then:
    # 1. All generated pairs use diseases from the training set only
    train_diseases = set(known_pairs_split[known_pairs_split["split"] == "TRAIN"]["target"])
    assert set(unknown["target"]).issubset(train_diseases)
    assert len(train_diseases) == expected_train_diseases

    # 2. No generated pairs exist in the known pairs set
    assert pd.merge(known_pairs, unknown, how="inner", on=["source", "target"]).empty

    # 3. All generated pairs are valid drug-disease pairs
    assert set(unknown["source"].to_list()).issubset(graph._drug_nodes)
    assert set(unknown["target"].to_list()).issubset(graph._disease_nodes)


def test_disease_split_drug_disease_pair_generator_no_train_diseases(graph: KnowledgeGraph, known_pairs: pd.DataFrame):
    """Test that DiseaseSplitDrugDiseasePairGenerator raises error when no training diseases are available."""
    # Given a split where all diseases are in test set
    splitter = ShuffleSplit(n_splits=1, test_size=0.66, random_state=111)
    known_pairs_split = make_folds(known_pairs, splitter)
    known_pairs_split = known_pairs_split[known_pairs_split["fold"] == 0]

    # Manually set all pairs to test set to ensure no training diseases
    known_pairs_split["split"] = "TEST"

    generator = DiseaseSplitDrugDiseasePairGenerator(
        random_state=42,
        n_unknown=10,
        y_label=2,
        drug_flags=["is_drug"],
        disease_flags=["is_disease"],
    )

    # When generating unknown pairs, it should raise ValueError
    with pytest.raises(ValueError, match="No training diseases found in the knowledge graph"):
        generator.generate(graph, known_pairs_split)


def test_disease_split_drug_disease_pair_generator_disease_distribution(
    graph: KnowledgeGraph, known_pairs: pd.DataFrame
):
    """Test that DiseaseSplitDrugDiseasePairGenerator maintains disease distribution in negative samples
    and ensures no test diseases appear in the training set.
    """
    # Given a train-test split and the generator
    splitter = ShuffleSplit(n_splits=1, test_size=0.5, random_state=111)
    known_pairs_split = make_folds(known_pairs, splitter)
    known_pairs_split = known_pairs_split[known_pairs_split["fold"] == 0]

    generator = DiseaseSplitDrugDiseasePairGenerator(
        random_state=42,
        n_unknown=100,  # Generate more samples for better distribution testing
        y_label=2,
        drug_flags=["is_drug"],
        disease_flags=["is_disease"],
    )

    # When generating unknown pairs
    unknown = generator.generate(graph, known_pairs_split)

    # Then:
    # 1. All diseases in the generated pairs are from training set
    train_diseases = set(known_pairs_split[known_pairs_split["split"] == "TRAIN"]["target"])
    test_diseases = set(known_pairs_split[known_pairs_split["split"] == "TEST"]["target"])
    assert set(unknown["target"]).issubset(train_diseases), "Test diseases found in generated pairs"

    # 2. Each training disease appears in the generated pairs
    disease_counts = unknown["target"].value_counts()
    assert set(disease_counts.index) == train_diseases, "Not all training diseases are represented"

    # 3. The distribution is roughly uniform (allowing for some randomness)
    expected_count = len(unknown) / len(train_diseases)
    assert all(
        abs(count - expected_count) < expected_count * 0.5 for count in disease_counts
    ), "Disease distribution is not roughly uniform"

    # 4. Create a simulated training set by combining positives and negatives
    training_positives = known_pairs_split[known_pairs_split["split"] == "TRAIN"]
    training_set = pd.concat([training_positives, unknown], ignore_index=True)

    # 5. Verify no test diseases in any part of the training set
    all_training_diseases = set(training_set["target"])
    test_diseases_in_training = test_diseases.intersection(all_training_diseases)
    assert not test_diseases_in_training, f"Test diseases found in training set: {test_diseases_in_training}"

    # 6. Verify no test diseases in negative samples specifically
    negative_training_diseases = set(training_set[training_set["y"] == 2]["target"])
    test_diseases_in_negatives = test_diseases.intersection(negative_training_diseases)
    assert not test_diseases_in_negatives, f"Test diseases found in negative samples: {test_diseases_in_negatives}"

    # 7. Print distribution information for debugging
    print("\nDisease Distribution in Training Set:")
    print(f"Total training diseases: {len(train_diseases)}")
    print(f"Total test diseases: {len(test_diseases)}")
    print(f"Training diseases in negatives: {len(negative_training_diseases)}")
    print("\nDisease counts in negative samples:")
    print(disease_counts)


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
    df = pd.DataFrame(data, index=np.random.permutation(100) + 1).sort_values("treat_score", ascending=False)

    # Ensure no overlap between positive and negative pairs
    all_cols = ["is_positive_1", "is_positive_2", "is_negative_1", "is_negative_2"]
    for col in all_cols:
        other_cols = [c for c in all_cols if c != col]
        df.loc[df[col], other_cols] = False

    df["rank"] = range(1, len(df) + 1)
    df["quantile_rank"] = df["rank"] / len(df)
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
    # Given a matrix and a test data generator with no removal columns
    generator_1 = MatrixTestDiseases(
        positive_columns=["is_positive_1", "is_positive_2"],
    )

    # When generating the test dataset
    generated_data_1 = generator_1.generate(matrix)

    # Then generated test data:
    #   - has the expected number of rows (# test diseases x # drugs)
    #   - has the expected number of positive pairs
    #   - has the expected columns
    positive_pairs = matrix[matrix["is_positive_1"] | matrix["is_positive_2"]]
    positive_diseases = positive_pairs["target"].unique()

    assert len(generated_data_1) == len(positive_diseases) * len(matrix["source"].unique())
    assert generated_data_1["y"].sum() == len(positive_pairs)
    assert set(generated_data_1.columns) == set(matrix.columns).union({"y"})

    # Given a matrix and a test data generator with a removal columns
    generator_2 = MatrixTestDiseases(
        positive_columns=["is_positive_1", "is_positive_2"],
        removal_columns=["is_negative_1"],
    )

    # When generating the test dataset
    generated_data_2 = generator_2.generate(matrix)

    # Then generated test data:
    #   - contains all rows where the target is in the positive pairs set
    negative_pairs = matrix[matrix["is_negative_1"]]
    negative_pairs_set = set(zip(negative_pairs["source"], negative_pairs["target"]))
    generated_data_set = set(zip(generated_data_2["source"], generated_data_2["target"]))

    assert len(generated_data_set.intersection(negative_pairs_set)) == 0


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
