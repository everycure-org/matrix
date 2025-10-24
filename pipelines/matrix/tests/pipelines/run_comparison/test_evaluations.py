from math import log

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matrix.pipelines.run_comparison.evaluations import (
    ComparisonEvaluationModelSpecific,
    EntropyAtN,
    FullMatrixRecallAtN,
    SpecificHitAtK,
)
from polars.testing import assert_frame_equal


@pytest.fixture
def constant_score_data():
    # Generate predictions
    N = 10
    combined_predictions = {
        "model_1_fold_0": lambda: pl.LazyFrame(
            {
                "source": list(range(N)),
                "target": list(range(N)),
                "score": 3 / 4 * np.ones(N),
                "is_known_positive": np.array([True] + [False for _ in range(N - 1)]),  # True in first position only
            },
        ),
        "model_1_fold_1": lambda: pl.LazyFrame(
            {
                "source": list(range(N)),
                "target": list(range(N)),
                "score": 1 / 2 * np.ones(N),
                "is_known_positive": np.array([False for _ in range(N - 1)] + [True]),  # True in last position only
            },
        ),
        "model_2_fold_0": lambda: pl.LazyFrame(
            {
                "source": list(range(N)),
                "target": list(range(N)),
                "score": 1 / 2 * np.ones(N),
                "is_known_positive": np.array([True] + [False for _ in range(N - 1)]),  # True in first position only
            },
        ),
        "model_2_fold_1": lambda: pl.LazyFrame(
            {
                "source": list(range(N)),
                "target": list(range(N)),
                "score": 1 / 4 * np.ones(N),
                "is_known_positive": np.array([False for _ in range(N - 1)] + [True]),  # True in last position only
            },
        ),
    }

    # Generate additional information
    predictions_info = {
        "model_names": ["model_1", "model_2"],
        "num_folds": 2,
        "available_ground_truth_cols": ["is_known_positive"],
    }
    return combined_predictions, predictions_info


class TestComparisonEvaluationModelSpecific(ComparisonEvaluationModelSpecific):
    """A class to test the concrete methods of the abstract class ComparisonEvaluationModelSpecific."""

    def give_x_values(self) -> np.ndarray:
        return np.array([0, 1])

    def give_y_values(self, matrix: pl.DataFrame) -> np.ndarray:
        # Return constant y value equal to the mean score
        return matrix["score"].mean() * np.ones(2)

    def give_y_values_bootstrap(self, matrix: pl.DataFrame) -> np.ndarray:
        # Return constant y value equal to the mean score plus/minus 1/4
        mean_score_curve = self.give_y_values(matrix)
        return np.array([mean_score_curve + 1 / 4, mean_score_curve - 1 / 4])

    def give_y_values_baseline(self, combined_predictions: dict[str, pl.LazyFrame]) -> np.ndarray:
        # Return constant zero values
        return np.zeros(2)


def test_model_specific_abstract_class(constant_score_data):
    """Test the abstract class ComparisonEvaluationModelSpecific."""
    # Given constant score data and an instance of a test subclass of ComparisonEvaluationModelSpecific
    combined_predictions, predictions_info = constant_score_data
    evaluation = TestComparisonEvaluationModelSpecific(
        x_axis_label="x",
        y_axis_label="y",
        title="Test Title",
        force_full_y_axis=False,
    )

    # When the concrete methods are called
    single_fold_results = evaluation.evaluate_single_fold(combined_predictions, predictions_info)
    multi_fold_results = evaluation.evaluate_multi_fold(combined_predictions, predictions_info)
    bootstrap_single_fold_results = evaluation.evaluate_bootstrap_single_fold(combined_predictions, predictions_info)
    bootstrap_multi_fold_results = evaluation.evaluate_bootstrap_multi_fold(combined_predictions, predictions_info)
    figure = evaluation.plot_results(
        single_fold_results, combined_predictions, predictions_info, perform_multifold=False, perform_bootstrap=False
    )

    # Then results are as expected
    # Single fold results take first fold as default
    assert_frame_equal(
        single_fold_results,
        pl.DataFrame({"x": [0, 1], "y_model_1": [3 / 4, 3 / 4], "y_model_2": [1 / 2, 1 / 2]}),
        check_row_order=False,
        check_column_order=False,
    )
    # Multi fold results take mean and std of all folds
    assert_frame_equal(
        multi_fold_results,
        pl.DataFrame(
            {
                "x": [0, 1],
                "y_model_1_mean": [5 / 8, 5 / 8],  # Mean of 3/4 and 1/2
                "y_model_1_std": [1 / 8, 1 / 8],  # Standard deviation of 3/4 and 1/2
                "y_model_2_mean": [3 / 8, 3 / 8],  # Mean of 1/2 and 1/4
                "y_model_2_std": [1 / 8, 1 / 8],  # Standard deviation of 1/2 and 1/4
            }
        ),
        check_row_order=False,
        check_column_order=False,
    )
    # Bootstrap single fold results take mean and std of bootstrap samples for first fold
    assert_frame_equal(
        bootstrap_single_fold_results,
        pl.DataFrame(
            {
                "x": [0, 1],
                "y_model_1_mean": [3 / 4, 3 / 4],  # Mean of 3/4+1/4 and 3/4-1/4
                "y_model_1_std": [1 / 4, 1 / 4],  # Standard deviation of 3/4+1/4 and 3/4-1/4
                "y_model_2_mean": [1 / 2, 1 / 2],  # Mean of 1/2+1/4 and 1/2-1/4
                "y_model_2_std": [1 / 4, 1 / 4],  # Standard deviation of 1/2+1/4 and 1/2-1/4
            }
        ),
        check_row_order=False,
        check_column_order=False,
    )
    # Bootstrap multi fold results take mean and std of bootstrap samples for all folds
    std = np.sqrt((2 * (1 / 8) ** 2 + 2 * (3 / 8) ** 2) / 4)
    assert_frame_equal(
        bootstrap_multi_fold_results,
        pl.DataFrame(
            {
                "x": [0, 1],
                "y_model_1_mean": [5 / 8, 5 / 8],  # Mean of [3/4+1/4, 3/4-1/4, 1/2+1/4, 1/2-1/4]
                "y_model_1_std": [std, std],  # Standard deviation of [3/4+1/4, 3/4-1/4, 1/2+1/4, 1/2-1/4]
                "y_model_2_mean": [3 / 8, 3 / 8],  # Mean of [1/2+1/4, 1/2-1/4, 1/4+1/4, 1/4-1/4]
                "y_model_2_std": [std, std],  # Standard deviation of [1/2+1/4, 1/2-1/4, 1/4+1/4, 1/4-1/4]
            }
        ),
        check_row_order=False,
        check_column_order=False,
    )
    # Plot results are as expected
    # Check figure is a matplotlib Figure object
    assert isinstance(figure, plt.Figure)
    assert figure is not None
    assert figure.get_axes()[0].get_xlabel() == "x"
    assert figure.get_axes()[0].get_ylabel() == "y"
    assert figure.get_axes()[0].get_title() == "Test Title"
    assert figure.get_axes()[0].get_legend().get_texts()[0].get_text() == "model_1"
    assert figure.get_axes()[0].get_legend().get_texts()[1].get_text() == "model_2"
    assert figure.get_axes()[0].get_legend().get_texts()[2].get_text() == "Random classifier"


# FullMatrixRecallAtN


@pytest.fixture
def matrix_data():
    return pl.DataFrame(
        {
            "source": [0, 0, 1, 1],
            "target": [0, 1, 0, 1],
            "is_known_positive": [False, True, True, False],
            "score": [1.0, 0.75, 0.5, 0.25],
        }
    )


def test_full_matrix_recall_at_n(matrix_data):
    """Test the FullMatrixRecallAtN class."""
    # Given matrix data, combined pairs data and an instance of FullMatrixRecallAtN
    matrix = matrix_data
    combined_pairs = {
        "model_fold_0": lambda: pl.LazyFrame(matrix)
    }  # Dummy function to simulate a Kedro Partitioned dataset
    evaluation = FullMatrixRecallAtN(
        ground_truth_col="is_known_positive",
        n_max=4,
        num_n_values=4,
        N_bootstraps=10,
        perform_sort=True,
        title="Test Title",
    )

    # When the method of the class are called
    x_values = evaluation.give_x_values()
    y_values = evaluation.give_y_values(matrix, "score")
    y_values_bootstrap = evaluation.give_y_values_bootstrap(matrix, "score")
    y_values_random = evaluation.give_y_values_baseline(combined_pairs)

    # Then the results are as expected
    # x_values are as expected
    assert np.allclose(x_values, np.array([1, 2, 3, 4]))
    # y_values are as expected
    assert np.allclose(y_values, np.array([0, 0.5, 1, 1]))
    # y_values_bootstrap have the right shape, and values are as expected
    assert y_values_bootstrap.shape == (10, 4)
    assert all(y_values_bootstrap[i][0] == 0 for i in range(10))  # Recall@1 always 0 regardless of sample
    assert all(
        (y_values_bootstrap[i][1] >= 0) and (y_values_bootstrap[i][1] <= 1) for i in range(10)
    )  # Recall@2 can be any value between 0 and 1
    assert all(
        (y_values_bootstrap[i][2] >= 0.5) and (y_values_bootstrap[i][2] <= 1) for i in range(10)
    )  # Recall@3 always >= 1/2 regardless of sample
    assert all(y_values_bootstrap[i][3] == 1 for i in range(10))  # Recall@4 always 1 regardless of sample
    # y_values_random are as expected
    assert np.allclose(y_values_random, np.array([0.25, 0.5, 0.75, 1]))


# SpecificHitAtK


@pytest.fixture
def disease_specific_hit_at_k_data():
    return pl.DataFrame(
        {
            "source": [1, 2, 3, 1, 2, 3],
            "target": [0, 0, 0, 1, 1, 1],
            "is_known_positive": [False, True, True, False, True, False],
            "score": [1, 2, 3, 1, 2, 3],
        }
    )


def test_disease_specific_hit_at_k(disease_specific_hit_at_k_data):
    """Test the SpecificHitAtK class."""
    # Given sample predictions data, combined pairs data and instances of SpecificHitAtK for disease-specific and drug-specific ranking
    matrix_disease_specific = disease_specific_hit_at_k_data
    matrix_drug_specific = disease_specific_hit_at_k_data.rename({"target": "source", "source": "target"})
    combined_pairs = {
        "disease_specific_model_fold_0": lambda: pl.LazyFrame(matrix_disease_specific),
        "drug_specific_model_fold_0": lambda: pl.LazyFrame(matrix_drug_specific),
    }  # Dummy functions to simulate a Kedro Partitioned dataset
    evaluation_disease_specific = SpecificHitAtK(
        ground_truth_col="is_known_positive",
        k_max=4,  # Set k max larger than the number of drugs
        title="Test Title",
        specific_col="target",
        N_bootstraps=10,
    )
    evaluation_drug_specific = SpecificHitAtK(
        ground_truth_col="is_known_positive",
        k_max=4,
        title="Test Title",
        specific_col="source",
        N_bootstraps=10,
    )

    # When the method of the classes are called
    x_values_disease_specific = evaluation_disease_specific.give_x_values()
    y_values_disease_specific = evaluation_disease_specific.give_y_values(matrix_disease_specific, "score")
    y_values_bootstrap_disease_specific = evaluation_disease_specific.give_y_values_bootstrap(
        matrix_disease_specific, "score"
    )
    y_values_random_disease_specific = evaluation_disease_specific.give_y_values_baseline(combined_pairs)
    y_values_drug_specific = evaluation_drug_specific.give_y_values(matrix_drug_specific, "score")

    # Then the results are as expected
    assert np.allclose(x_values_disease_specific, np.array([0, 1, 2, 3, 4]))
    # NOTE: The two drugs for disease 0 have rank 1 against negatives. The one drug for disease 1 has rank 2 against negatives.
    assert np.allclose(y_values_disease_specific, np.array([0, 2 / 3, 1, 1, 1]))
    assert y_values_bootstrap_disease_specific.shape == (10, 5)
    assert all(y_values_bootstrap_disease_specific[i][0] == 0 for i in range(10))  # Hit@0 always 0 regardless of sample
    assert all(
        (y_values_bootstrap_disease_specific[i][1] >= 0) and (y_values_bootstrap_disease_specific[i][1] <= 1)
        for i in range(10)
    )  # Hit@1 can be any value between 0 and 1
    assert all(
        y_values_bootstrap_disease_specific[i][j] == 1 for i in range(10) for j in [2, 3, 4]
    )  # Hit@2, Hit@3 and Hit@4 always 1 regardless of sample
    assert np.allclose(y_values_random_disease_specific, np.array([0, 1 / 3, 2 / 3, 1, 1]))
    assert np.allclose(
        y_values_drug_specific, np.array([0, 2 / 3, 1, 1, 1])
    )  # Same as disease-specific since the source and target columns were swapped


# EntropyAtN


@pytest.fixture
def entropy_at_n_data_uniform():
    return pl.DataFrame(
        {
            "source": [1, 2, 3, 1, 2, 3, 1, 2],  # Top 3 and Top 6 drugs have a uniform count distribution
            "target": [1, 2, 3, 4, 1, 2, 3, 4],  # Top 4 and Top 8 diseases have a uniform count distribution
            "is_known_positive": [False] * 8,
            "score": [1 / (i + 1) for i in range(8)],  # Scores are ordered in descending order
        }
    )


@pytest.fixture
def entropy_at_n_data_skewed():
    return pl.DataFrame(
        {
            "source": [1, 1, 1, 1] + [2, 3],  # Top 4 contains only one drug out of 3
            "target": [1, 2, 1, 2] + [2, 3],  # Top 4 contains uniformly distributed 2 diseases out of 3
            "is_known_positive": [False] * 6,
            "score": [1 / (i + 1) for i in range(6)],  # Scores are ordered in descending order
        }
    )


def test_entropy_at_n(entropy_at_n_data_uniform, entropy_at_n_data_skewed):
    # Given sample predictions data, combined pairs data and instances of EntropyAtN
    combined_pairs = {
        "model_fold_0": lambda: pl.LazyFrame(entropy_at_n_data_uniform),
    }  # Dummy functions to simulate a Kedro Partitioned dataset
    evaluation_drug_entropy = EntropyAtN(
        count_col="source",
        n_max=8,
        perform_sort=True,
        title="Drug-Entropy@n",
        num_n_values=9,
        force_full_y_axis=True,
    )
    evaluation_disease_entropy = EntropyAtN(
        count_col="target",
        n_max=8,
        perform_sort=True,
        title="Disease-Entropy@n",
        num_n_values=9,
        force_full_y_axis=True,
    )

    # When the method of the class are called
    x_values_drug = evaluation_drug_entropy.give_x_values()
    y_values_drug_uniform = evaluation_drug_entropy.give_y_values(entropy_at_n_data_uniform)
    y_values_disease_uniform = evaluation_disease_entropy.give_y_values(entropy_at_n_data_uniform)
    y_values_drug_skewed = evaluation_drug_entropy.give_y_values(entropy_at_n_data_skewed)
    y_values_disease_skewed = evaluation_disease_entropy.give_y_values(entropy_at_n_data_skewed)
    y_values_bootstraps_drug_uniform = evaluation_drug_entropy.give_y_values_bootstrap(entropy_at_n_data_uniform)
    y_values_baseline_drug = evaluation_drug_entropy.give_y_values_baseline(combined_pairs)

    # Then the results are as expected
    # Checking x values
    assert list(x_values_drug) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # Bootstrap values the same as no bootstraps
    assert np.allclose(y_values_bootstraps_drug_uniform, y_values_drug_uniform, atol=1e-6)
    # Checking y values for uniformly distributed data
    assert np.allclose(y_values_drug_uniform[3], 1, atol=1e-6)  # All 3 drugs appear once in top 3 so Entropy@3 = 1
    assert np.allclose(y_values_drug_uniform[6], 1, atol=1e-6)  # All 3 drugs appear twice in top 6 so Entropy@6 = 1
    assert np.allclose(
        y_values_disease_uniform[4], 1, atol=1e-6
    )  # All 4 diseases appear once in top 4 so Entropy@4 = 1
    assert np.allclose(
        y_values_disease_uniform[8], 1, atol=1e-6
    )  # All 4 diseases appear twice in top 8 so Entropy@8 = 1
    # Checking y values for skewed data
    assert np.allclose(
        y_values_drug_skewed[4], 0, atol=1e-6
    )  # Only one drug appears in top 4 so Entropy@4 = log_(1) = 0
    assert np.allclose(
        y_values_disease_skewed[4], log(2, 3), atol=1e-6
    )  # 2 diseases out of 3 appear uniformly in top 4 so Entropy@4 = log_3(2)
    # Checking baseline values
    assert np.allclose(
        y_values_baseline_drug[3], 1, atol=1e-6
    )  # 3 drugs can uniformly fill top 3 so Entropy@3 = 1 for maximum entropy
    assert np.allclose(
        y_values_baseline_drug[6], 1, atol=1e-6
    )  # 3 drugs can uniformly fill top 6 so Entropy@6 = 1 for maximum entropy
