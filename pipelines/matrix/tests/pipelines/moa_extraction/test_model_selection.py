import pandas as pd
import pytest
from matrix.pipelines.moa_extraction.model_selection import GroupAwareSplit


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "source": ["drug1", "drug1", "drug2", "drug2", "drug3"],
            "target": ["disease1", "disease2", "disease3", "disease4", "disease5"],
        }
    )


def test_group_aware_split(sample_data):
    # Given a GroupAwareSplit instance
    splitter = GroupAwareSplit(group_by_column="source", n_splits=1, test_size=0.1, random_state=42)

    # When we split the data
    splits = list(splitter.split(sample_data))

    # Then the result has the correct number of splits
    assert len(splits) == 1

    # And the splits have the correct number of samples
    train_indices, test_indices = splits[0]
    assert len(train_indices) + len(test_indices) == len(sample_data)

    # And the test and train have no overlapping values for the "group_by" column
    train_sources = set(sample_data.iloc[train_indices]["source"])
    test_sources = set(sample_data.iloc[test_indices]["source"])
    assert len(train_sources.intersection(test_sources)) == 0
