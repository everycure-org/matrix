import pandas as pd
import pytest
from matrix.pipelines.modelling.model_selection import DiseaseAreaSplit, DrugCVSplit, DrugStratifiedSplit
from sklearn.model_selection import StratifiedShuffleSplit


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "source": ["drug1", "drug1", "drug2", "drug2", "drug3"] * 5,
            "target": ["disease1", "disease2", "disease3", "disease4", "disease5"] * 5,
            "y": [1, 0, 1, 0, 1] * 5,
        }
    )


@pytest.fixture
def disease_list_data():
    return pd.DataFrame(
        {
            "id": ["disease1", "disease2", "disease3", "disease4", "disease5"],
            "harrisons_view": [
                "cancer_or_benign_tumor",
                "hereditary_disease",
                "inflammatory_disease",
                "metabolic_disease",
                "syndromic_disease",
            ],
            "mondo_top_grouping": [
                "infectious_disease",
                "infectious_disease",
                "cardiovascular_disorder",
                "cardiovascular_disorder",
                "nervous_system_disorder",
            ],
            "mondo_txgnn": [
                "cancer_or_benign_tumor",
                "cancer_or_benign_tumor",
                "infectious_disease",
                "infectious_disease",
                "psychiatric_disorder",
            ],
            "txgnn": [
                "mental_health_disorder",
                "mental_health_disorder",
                "cancer",
                "cancer",
                "cardiovascular_disorder",
            ],
        }
    )


def test_drug_stratified_split(sample_data):
    # Given a splitter and a drug disease dataset
    splitter = DrugStratifiedSplit(n_splits=1, test_size=0.1, random_state=42)

    # When we split the data
    splits = list(splitter.split(sample_data))

    # Then the result is of the correct format and has the correct number of test and train pairs
    assert len(splits) == 1
    train_indices, test_indices = splits[0]
    assert len(train_indices) + len(test_indices) == len(sample_data)

    # Check that the split is stratified by drug
    train_drugs = set(sample_data.iloc[train_indices]["source"])
    test_drugs = set(sample_data.iloc[test_indices]["source"])
    assert len(train_drugs.intersection(test_drugs)) == 3

    # Check that train and test sets are unique
    assert len(set(train_indices).intersection(set(test_indices))) == 0


def test_stratified_shuffle_split(sample_data):
    # Given a StratifiedShuffleSplit and a drug disease dataset
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

    # When we split the data
    splits = list(splitter.split(sample_data, sample_data["y"]))

    # Then the result is of the correct format and has the correct number of test and train pairs
    assert len(splits) == 1
    train_indices, test_indices = splits[0]
    assert len(train_indices) + len(test_indices) == len(sample_data)

    # Check that the split is stratified by y
    train_y = sample_data.iloc[train_indices]["y"]
    test_y = sample_data.iloc[test_indices]["y"]
    assert set(train_y) == set(test_y) == {0, 1}


def test_drug_stratified_split_multiple_splits(sample_data):
    # Given a DrugStratifiedSplit instance with multiple splits
    splitter = DrugStratifiedSplit(n_splits=3, test_size=0.1, random_state=42)

    # When we split the data
    splits = list(splitter.split(sample_data))

    # Then the result is of the correct format and has the correct number of test and train pairs
    assert len(splits) == 3
    for train_indices, test_indices in splits:
        assert len(train_indices) + len(test_indices) == len(sample_data)

        train_drugs = set(sample_data.iloc[train_indices]["source"])
        test_drugs = set(sample_data.iloc[test_indices]["source"])
        assert len(train_drugs.intersection(test_drugs)) > 0


def test_disease_area_split_harrisons_view(sample_data, disease_list_data):
    # Given a DiseaseAreaSplit instance with harrisons_view disease grouping
    holdout_disease_types = ["cancer_or_benign_tumor", "inflammatory_disease", "hereditary_disease"]
    splitter = DiseaseAreaSplit(
        n_splits=2,
        test_size=0.1,
        random_state=42,
        disease_grouping_type="harrisons_view",
        holdout_disease_types=holdout_disease_types,
    )

    # When we split the data
    splits = list(splitter.split(sample_data, disease_list_data))

    # Then the result is of the correct format and has the correct number of splits
    assert len(splits) == 2

    # Check that the internal operations within DiseaseAreaSplit are correct
    for i, (train_indices, test_indices) in enumerate(splits):
        merged_data = sample_data.merge(
            disease_list_data[["id", "harrisons_view"]],
            left_on="target",
            right_on="id",
            how="left",
        )
        matched_data = merged_data[~merged_data.id.isna()]
        assert len(train_indices) + len(test_indices) == len(matched_data)

        # Check that train and test sets are disjoint
        assert len(set(train_indices).intersection(set(test_indices))) == 0

        # Check that test set contains only diseases of selected holdout type
        test_data = sample_data.iloc[test_indices]
        test_diseases = set(test_data["target"])

        # Find the selected disease type for this split
        selected_disease_type = holdout_disease_types[i]

        # Verify all test diseases are of the selected holdout type
        for disease in test_diseases:
            disease_type = disease_list_data[disease_list_data.id == disease]["harrisons_view"].iloc[0]
            assert disease_type == selected_disease_type


def test_disease_area_split_invalid_grouping(sample_data, disease_list_data):
    # Given a DiseaseAreaSplit with an invalid disease grouping type
    splitter = DiseaseAreaSplit(
        n_splits=1, disease_grouping_type="invalid_grouping", holdout_disease_types=["some_type"]
    )

    # When we try to split the data, it should raise a ValueError
    with pytest.raises(ValueError, match="Disease grouping type invalid_grouping not found in disease_list"):
        next(splitter.split(sample_data, disease_list_data))


def test_disease_area_split_insufficient_holdout_types():
    # Given a DiseaseAreaSplit with fewer holdout disease types than n_splits
    with pytest.raises(ValueError, match="Length of holdout_disease_types is less than n_splits"):
        DiseaseAreaSplit(n_splits=3, disease_grouping_type="harrisons_view", holdout_disease_types=["type1", "type2"])


def test_drug_cv_split(sample_data):
    # Given a DrugCVSplit and a drug disease dataset
    splitter = DrugCVSplit(n_splits=1, test_size=0.2, random_state=42)

    # When we split the data
    splits = list(splitter.split(sample_data))

    # Then the result is of the correct format and has the correct number of test and train pairs
    assert len(splits) == 1
    train_indices, test_indices = splits[0]
    assert len(train_indices) + len(test_indices) == len(sample_data)

    # Check that the split is by drug (no drug appears in both train and test)
    train_drugs = set(sample_data.iloc[train_indices]["source"])
    test_drugs = set(sample_data.iloc[test_indices]["source"])
    assert len(train_drugs.intersection(test_drugs)) == 0

    # Check that train and test sets are unique
    assert len(set(train_indices).intersection(set(test_indices))) == 0

    # Check that approximately the right proportion of drugs are in the test set
    unique_drugs = set(sample_data["source"].unique())
    expected_test_drugs = max(1, int(round(len(unique_drugs) * 0.2)))
    assert len(test_drugs) == expected_test_drugs


def test_drug_cv_split_multiple_splits(sample_data):
    # Given a DrugCVSplit instance with multiple splits
    splitter = DrugCVSplit(n_splits=3, test_size=0.2, random_state=42)

    # When we split the data
    splits = list(splitter.split(sample_data))

    # Then the result is of the correct format and has the correct number of test and train pairs
    assert len(splits) == 3
    for train_indices, test_indices in splits:
        assert len(train_indices) + len(test_indices) == len(sample_data)

        # Check that the split is by drug (no drug appears in both train and test)
        train_drugs = set(sample_data.iloc[train_indices]["source"])
        test_drugs = set(sample_data.iloc[test_indices]["source"])
        assert len(train_drugs.intersection(test_drugs)) == 0

        # Check that approximately the right proportion of drugs are in the test set
        unique_drugs = set(sample_data["source"].unique())
        expected_test_drugs = max(1, int(round(len(unique_drugs) * 0.2)))
        assert len(test_drugs) == expected_test_drugs
