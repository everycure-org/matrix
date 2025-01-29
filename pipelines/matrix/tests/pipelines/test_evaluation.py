import numpy as np
import pandas as pd
import pytest
from matrix.datasets.pair_generator import DrugDiseasePairGenerator
from matrix.pipelines.evaluation.evaluation import (
    ContinuousMetrics,
    DiscreteMetrics,
    FullMatrixRanking,
    RecallAtN,
    SpecificRanking,
)
from matrix.pipelines.evaluation.named_metric_functions import (
    AUROC,
    MRR,
    CommonalityAtN,
    HitK,
    HypergeomAtN,
    SpearmanAtN,
)
from matrix.pipelines.evaluation.named_metric_functions import RecallAtN as RecallAtN_
from matrix.pipelines.evaluation.nodes import generate_test_dataset
from sklearn.metrics import accuracy_score, roc_auc_score


@pytest.fixture
def sample_data():
    """Fixture that provides sample DataFrames for testing evaluation metrics."""
    standard_case = pd.DataFrame(
        {
            "source": ["A", "A", "B", "B", "C"],
            "target": [1, 2, 1, 2, 1],
            "y": [1, 0, 1, 0, 1],
            "score": [0.7, 0.3, 0.6, 0.4, 0.8],
        }
    )

    edge_case = pd.DataFrame(
        {
            "source": ["A", "A", "B", "B", "C"],
            "target": [1, 2, 3, 4, 5],
            "y": [1, 1, 1, 1, 1],  # All positive labels
            "score": [0.5, 0.5, 0.5, 0.5, 0.5],  # All same scores
        }
    )

    return standard_case, edge_case


@pytest.fixture
def sample_positives():
    """Fixture that provides sample DataFrames for testing FullMatrixPositives."""
    data = pd.DataFrame(
        {
            "source": [
                "A",
                "A",
                "B",
            ],
            "target": [1, 2, 1],
            "y": [1, 1, 1],
            # Suppose that there are 3 positives and 2 non-positive pairs in the matrix
            "rank": [1, 2, 5],
            "non_pos_rank": [1, 1, 3],
        }
    )
    data["non_pos_quantile_rank"] = (data["non_pos_rank"] - 1) / 2
    return data


@pytest.fixture
def sample_matrix():
    """Sample matrix dataframe fixture."""
    return pd.DataFrame(
        {
            "source": ["drug1", "drug2", "drug3"],
            "target": ["disease1", "disease2", "disease3"],
            "score": [0.9, 0.8, 0.7],
        }
    )


@pytest.fixture
def mock_generator():
    """Mock generator that returns predefined test data."""

    class MockGenerator(DrugDiseasePairGenerator):
        def generate(self, matrix):
            return pd.DataFrame({"source": ["drug1", "drug2"], "target": ["disease1", "disease2"], "y": [1, 0]})

    return MockGenerator()


def test_discrete_metrics(sample_data):
    """Test the DiscreteMetrics class using accuracy as the metric."""
    # Given standard and edge case datasets, and accuracy as the evaluation metric
    standard_case, edge_case = sample_data
    metrics = [accuracy_score]
    evaluator = DiscreteMetrics(metrics, score_col_name="score", threshold=0.5)

    # When DiscreteMetrics is initialized with a threshold of 0.5 and evaluates both datasets
    standard_result = evaluator.evaluate(standard_case)
    edge_result = evaluator.evaluate(edge_case)

    # Then the accuracy scores should be correctly calculated for both cases
    assert "accuracy_score" in standard_result
    assert standard_result["accuracy_score"] == 1.0

    assert "accuracy_score" in edge_result
    assert edge_result["accuracy_score"] == 1.0


def test_continuous_metrics(sample_data):
    """Test the ContinuousMetrics class using ROC AUC as the metric."""
    # Given standard and edge case datasets, and ROC AUC as the evaluation metric
    standard_case, edge_case = sample_data
    metrics = [roc_auc_score]
    evaluator = ContinuousMetrics(metrics, score_col_name="score")

    # When ContinuousMetrics evaluates both datasets
    standard_result = evaluator.evaluate(standard_case)
    edge_result = evaluator.evaluate(edge_case)

    # Then the ROC AUC scores should be correctly calculated for both cases
    assert "roc_auc_score" in standard_result
    assert standard_result["roc_auc_score"] == 1.0

    assert "roc_auc_score" in edge_result
    assert edge_result["roc_auc_score"] == 0.5  # ROC AUC is undefined for all same labels, defaults to 0.5


def test_specific_ranking(sample_data):
    """Test the SpecificRanking class using MRR and Hit@1 as ranking functions."""
    # Given standard and edge case datasets, and MRR and Hit@1 as ranking metrics
    standard_case, edge_case = sample_data
    rank_funcs = [MRR(), HitK(k=1)]
    evaluator = SpecificRanking(rank_funcs, specific_col="source", score_col_name="score")

    # When SpecificRanking evaluates both datasets with source-specific ranking
    standard_result = evaluator.evaluate(standard_case)
    edge_result = evaluator.evaluate(edge_case)

    # Then both MRR and Hit@1 scores should be correctly calculated for both cases
    assert "mrr" in standard_result and "hit-1" in standard_result
    assert standard_result["mrr"] == 1.0
    assert standard_result["hit-1"] == 1.0

    assert "mrr" in edge_result and "hit-1" in edge_result
    assert edge_result["mrr"] == 1.0
    assert edge_result["hit-1"] == 1.0


def test_full_matrix_ranking(sample_positives):
    """Test the FullMatrixRanking class using RecallAtN and AUROC metrics."""
    # Given a sample dataset that is the output of FullMatrixPositives
    data = sample_positives
    rank_funcs = [RecallAtN_(n=2), RecallAtN_(n=5)]
    quantile_funcs = [AUROC()]

    # When FullMatrixRanking is initialized with RecallAtN and AUROC metrics
    evaluator = FullMatrixRanking(rank_func_lst=rank_funcs, quantile_func_lst=quantile_funcs)

    # And when it evaluates the dataset
    result = evaluator.evaluate(data)

    # Then the RecallAtN and AUROC scores should be correctly calculated
    assert "recall-2" in result
    assert "recall-5" in result
    assert "auroc" in result

    # Verify RecallAtN calculations
    assert np.isclose(result["recall-2"], 2 / 3, atol=1e-6)  # 2 out of 3 true positives in top 2
    assert np.isclose(result["recall-5"], 1.0, atol=1e-6)  # All 3 true positives in top 5

    # Verify AUROC calculation
    assert np.isclose(result["auroc"], 2 / 3, atol=1e-6)


def test_recall_at_n(sample_data):
    """Test the RecallAtN class."""

    # Test with multiple N values
    recall_evaluator = RecallAtN(n_values=[3, 5], score_col_name="score")
    data_standard_case = sample_data[0]
    result = recall_evaluator.evaluate(data_standard_case)

    assert "recall_at_3" in result
    assert np.isclose(result["recall_at_3"], 3 / 3, atol=1e-6)  # 2 out of 3 true positives in top 3

    assert "recall_at_5" in result
    assert np.isclose(result["recall_at_5"], 1.0, atol=1e-6)  # All true positives included


def test_generate_test_dataset_injection():
    """Test object injection functionality."""
    sample_df = pd.DataFrame({"source": ["drug1"], "target": ["disease1"], "score": [0.9]})

    # Test with string-based injection
    with pytest.raises(Exception):
        # Should fail because DrugDiseasePairGenerator is abstract
        generate_test_dataset(
            matrix=sample_df, generator={"object": "matrix.datasets.pair_generator.DrugDiseasePairGenerator"}
        )


@pytest.fixture
def sample_rank_sets():
    """Fixture that provides sample rank sets for testing."""
    return (
        pd.DataFrame({"pair_id": [1, 2, 3, 4, 5], "rank": [12, 522, 33, 14, 1]}),
        pd.DataFrame({"pair_id": [1, 2, 3, 5, 6], "rank": [1, 2, 3, 4, 5]}),
    )


@pytest.fixture
def sample_hypergeom_sets():
    """Fixture that provides sample rank sets for testing."""
    return (
        pd.DataFrame({"pair_id": [1, 2, 3, 4, 5, 6], "rank": [1, 2, 3, 4, 5, 6]}),  # Increased to 6 items
        pd.DataFrame({"pair_id": [3, 4, 5, 7, 8, 9], "rank": [1, 2, 3, 4, 5, 6]}),  # Keep 5 items to ensure overlap
    )


@pytest.fixture
def sample_commonality_matrices():
    """Fixture that provides sample matrices for testing commonality."""
    return [pd.DataFrame({"pair_id": [1, 2, 3]}), pd.DataFrame({"pair_id": [1, 3, 5]})]


def test_hypergeom_at_n(sample_hypergeom_sets):
    """Test the HypergeomAtN class."""
    n = 5
    hypergeom_evaluator = HypergeomAtN(n)
    result = hypergeom_evaluator.generate()(sample_hypergeom_sets, common_items=pd.DataFrame({"pair_id": [3, 4, 5]}))

    result_same = hypergeom_evaluator.generate()(
        (sample_hypergeom_sets[0], sample_hypergeom_sets[0]), common_items=pd.DataFrame({"pair_id": [3, 4, 5]})
    )
    assert result["pvalue"] > 0.05


def test_spearman_at_n(sample_rank_sets):
    """Test the SpearmanAtN class."""
    n = 4
    spearman_evaluator = SpearmanAtN(n)

    # First one is testing the pvalue
    result = spearman_evaluator.generate()(sample_rank_sets, common_items=pd.DataFrame({"pair_id": [1, 2, 3, 4]}))

    # Second one is testing correlation
    result_same = spearman_evaluator.generate()(
        (sample_rank_sets[0], sample_rank_sets[0]), common_items=pd.DataFrame({"pair_id": [1, 2, 3, 4]})
    )

    assert result["pvalue"] > 0.05
    assert round(result_same["correlation"]) == 1.0
    assert result_same["pvalue"] < 0.05


def test_commonality_at_n(sample_commonality_matrices):
    """Test the CommonalityAtN class."""
    n = 3
    commonality_evaluator = CommonalityAtN(n)
    result = commonality_evaluator.generate()(sample_commonality_matrices)
    assert round(result, 2) == 0.67
