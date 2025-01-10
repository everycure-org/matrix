import pandas as pd
import pytest
from matrix.pipelines.modelling.model_selection import DrugStratifiedSplit
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
