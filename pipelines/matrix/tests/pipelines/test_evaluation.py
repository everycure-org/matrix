import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from matrix.pipelines.evaluation.evaluation import (
    DiscreteMetrics,
    ContinuousMetrics,
    SpecificRanking,
    MRR,
    HitK,
)


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
    assert (
        edge_result["roc_auc_score"] == 0.5
    )  # ROC AUC is undefined for all same labels, defaults to 0.5


def test_specific_ranking(sample_data):
    """Test the SpecificRanking class using MRR and Hit@1 as ranking functions."""
    # Given standard and edge case datasets, and MRR and Hit@1 as ranking metrics
    standard_case, edge_case = sample_data
    rank_funcs = [MRR(), HitK(k=1)]
    evaluator = SpecificRanking(
        rank_funcs, specific_col="source", score_col_name="score"
    )

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
