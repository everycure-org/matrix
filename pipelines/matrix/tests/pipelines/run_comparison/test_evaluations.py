import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matrix.pipelines.run_comparison.evaluations import ComparisonEvaluationModelSpecific, FullMatrixRecallAtN
from polars.testing import assert_frame_equal


@pytest.fixture
def constant_score_data():
    # Generate predictions
    N = 10
    predictions = pl.LazyFrame(
        {
            "source": list(range(N)),
            "target": list(range(N)),
            "score_model_1_fold_0": 3 / 4 * np.ones(N),
            "score_model_1_fold_1": 1 / 2 * np.ones(N),
            "score_model_2_fold_0": 1 / 2 * np.ones(N),
            "score_model_2_fold_1": 1 / 4 * np.ones(N),
            "is_known_positive_fold_0": np.array([True] + [False for _ in range(N - 1)]),  # True in first position only
            "is_known_positive_fold_1": np.array([False for _ in range(N - 1)] + [True]),  # True in last position only
        }
    )

    # Generate additional information
    predictions_info = {
        "model_names": ["model_1", "model_2"],
        "num_folds": 2,
        "available_ground_truth_cols": ["is_known_positive"],
    }
    return predictions, predictions_info


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

    def give_y_values_random_classifier(self, combined_predictions: pl.LazyFrame) -> np.ndarray:
        # Return constant zero values
        return np.zeros(2)


def test_model_specific_abstract_class(constant_score_data):
    """Test the abstract class ComparisonEvaluationModelSpecific."""
    # Given constant score data and an instance of a test subclass of ComparisonEvaluationModelSpecific
    predictions, predictions_info = constant_score_data
    evaluation = TestComparisonEvaluationModelSpecific(
        x_axis_label="x",
        y_axis_label="y",
        title="Test Title",
        force_full_y_axis=False,
    )

    # When the concrete methods are called
    single_fold_results = evaluation.evaluate_single_fold(predictions, predictions_info)
    multi_fold_results = evaluation.evaluate_multi_fold(predictions, predictions_info)
    bootstrap_single_fold_results = evaluation.evaluate_bootstrap_single_fold(predictions, predictions_info)
    bootstrap_multi_fold_results = evaluation.evaluate_bootstrap_multi_fold(predictions, predictions_info)
    figure = evaluation.plot_results(single_fold_results, predictions, predictions_info, is_plot_errors=False)

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


def test_model_full_matrix_recall_at_n(matrix_data):
    """Test the FullMatrixRecallAtN class."""
    # Given matrix data, combined predictions and an instance of FullMatrixRecallAtN
    matrix = matrix_data
    combined_predictions = pl.LazyFrame(matrix)
    evaluation = FullMatrixRecallAtN(
        ground_truth_col="is_known_positive",
        n_max=4,
        num_n_values=4,
        N_bootstraps=10,
        perform_sort=True,
        title="Test Title",
    )

    # When the evaluate_single_fold method is called
    x_values = evaluation.give_x_values()
    y_values = evaluation.give_y_values(matrix, "score")
    y_values_bootstrap = evaluation.give_y_values_bootstrap(matrix, "score")
    y_values_random = evaluation.give_y_values_random_classifier(combined_predictions)

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
