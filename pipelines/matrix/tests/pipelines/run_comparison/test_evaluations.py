import pandas as pd
import pytest
from matrix.pipelines.run_comparison.evaluations import ComparisonEvaluationModelSpecific


@pytest.fixture
def constant_score_data():
    # TODO: generate in the form f"score_{model_name}_fold_{fold}",
    N = 10
    pairs = pd.DataFrame(
        {
            "source": list(range(N)),
            "target": list(range(N)),
        }
    )
    matrix_1 = pairs.copy(deep=True)
    matrix_1["score"] = 3 / 4
    matrix_2 = pairs.copy(deep=True)
    matrix_2["score"] = 1 / 4
    return matrix_1, matrix_2


class TestComparisonEvaluationModelSpecific(ComparisonEvaluationModelSpecific):
    """A class to test the concrete methods of the abstract class ComparisonEvaluationModelSpecific."""

    def give_x_values(self) -> np.ndarray:
        return np.array([0, 1])

    def give_y_values(self, matrix: pd.DataFrame, score_col_name: str) -> np.ndarray:
        # Return constant y value equal to the mean score
        return matrix[score_col_name].mean() * np.ones(2)

    def give_y_values_bootstrap(self, matrix: pd.DataFrame, score_col_name: str) -> np.ndarray:
        # Return constant y value equal to the mean score plus/minus 1/4
        mean_score_curve = self.give_y_values(matrix, score_col_name)
        return np.array([mean_score_curve + 1 / 4, mean_score_curve - 1 / 4])

    def give_y_values_random_classifier(self, combined_predictions: pl.LazyFrame) -> np.ndarray:
        # Return constant y value equal to the mean score
        return np.zeros(2)


def test_model_specific_abstract_class(constant_score_data):
    """Test the abstract class ComparisonEvaluationModelSpecific."""
    # Given constant score data and an instance of a test subclass of ComparisonEvaluationModelSpecific
    matrix_1, matrix_2 = constant_score_data
    evaluation = TestComparisonEvaluationModelSpecific()

    # When the
