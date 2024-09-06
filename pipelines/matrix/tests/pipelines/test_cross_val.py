import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from matrix.pipelines.modelling.cross_val import DrugStratifiedSplit
from matrix.pipelines.modelling.nodes import make_splits

# Mock KnowledgeGraph class for testing
class MockKnowledgeGraph:
    def __init__(self):
        self._embeddings = {f"drug_{i}": np.random.rand(10) for i in range(10)}
        self._embeddings.update({f"disease_{i}": np.random.rand(10) for i in range(10)})

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'source': [f'drug_{i}' for i in range(10)] * 5,
        'target': [f'disease_{i}' for i in range(10)] * 5,
        'y': np.random.randint(0, 2, 50)
    })

@pytest.fixture
def mock_kg():
    return MockKnowledgeGraph()

def test_drug_stratified_split():
    splitter = DrugStratifiedSplit(n_splits=1, test_size=0.2, random_state=42)
    data = pd.DataFrame({
        'source': ['drug_1'] * 50 + ['drug_2'] * 50,
        'target': [f'disease_{i}' for i in range(100)],
        'y': np.random.randint(0, 2, 100)
    })

    result = splitter.split_with_labels(data)

    assert len(result) == len(data)
    assert 'iteration' in result.columns
    assert 'split' in result.columns
    assert set(result['split'].unique()) == {'TRAIN', 'TEST'}
    assert len(result[result['split'] == 'TRAIN']) == 80
    assert len(result[result['split'] == 'TEST']) == 20

    for drug in data['source'].unique():
        drug_data = result[result['source'] == drug]
        assert len(drug_data[drug_data['split'] == 'TRAIN']) == 40
        assert len(drug_data[drug_data['split'] == 'TEST']) == 10

def test_make_splits_with_drug_stratified_split(sample_data, mock_kg):
    splitter = DrugStratifiedSplit(n_splits=1, test_size=0.2, random_state=42)
    result = make_splits(mock_kg, sample_data, splitter)

    assert len(result) == len(sample_data)
    assert 'iteration' in result.columns
    assert 'split' in result.columns
    assert set(result['split'].unique()) == {'TRAIN', 'TEST'}
    assert len(result[result['split'] == 'TRAIN']) == 40
    assert len(result[result['split'] == 'TEST']) == 10

    for drug in sample_data['source'].unique():
        drug_data = result[result['source'] == drug]
        assert len(drug_data[drug_data['split'] == 'TRAIN']) == 4
        assert len(drug_data[drug_data['split'] == 'TEST']) == 1

    
def test_make_splits_with_stratified_shuffle_split(sample_data, mock_kg):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    result = make_splits(mock_kg, sample_data, splitter)

    assert len(result) == len(sample_data)
    assert 'iteration' in result.columns
    assert 'split' in result.columns
    assert set(result['split'].unique()) == {'TRAIN', 'TEST'}
    assert len(result[result['split'] == 'TRAIN']) == 40
    assert len(result[result['split'] == 'TEST']) == 10

    # Check if the stratification is maintained
    train_y_dist = result[result['split'] == 'TRAIN']['y'].value_counts(normalize=True)
    test_y_dist = result[result['split'] == 'TEST']['y'].value_counts(normalize=True)
    assert np.allclose(train_y_dist, test_y_dist, atol=0.1)  # Allow 10% tolerance

   
def test_make_splits_output_format(sample_data, mock_kg):
    splitter = DrugStratifiedSplit(n_splits=1, test_size=0.2, random_state=42)
    result = make_splits(mock_kg, sample_data, splitter)

    assert isinstance(result, pd.DataFrame)
    assert 'source' in result.columns
    assert 'target' in result.columns
    assert 'y' in result.columns
    assert 'iteration' in result.columns
    assert 'split' in result.columns
    assert 'source_embedding' in result.columns
    assert 'target_embedding' in result.columns

    assert result['split'].isin(['TRAIN', 'TEST']).all()
    assert result['iteration'].nunique() == 1

def test_make_splits_multiple_iterations(sample_data, mock_kg):
    splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    result = make_splits(mock_kg, sample_data, splitter)

    assert len(result) == len(sample_data) * 3  # 3 iterations
    assert 'iteration' in result.columns
    assert 'split' in result.columns
    assert set(result['split'].unique()) == {'TRAIN', 'TEST'}
    assert result['iteration'].nunique() == 3

    for iteration in range(3):
        iteration_data = result[result['iteration'] == iteration]
        assert len(iteration_data[iteration_data['split'] == 'TRAIN']) == 40
        assert len(iteration_data[iteration_data['split'] == 'TEST']) == 10
