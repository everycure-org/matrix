import polars as pl
import pytest


@pytest.fixture
def sample_data_inconsistent_models():
    return {
        "base_model": {
            "predictions_list": [
                pl.LazyFrame(
                    {
                        "source": [1, 2, 1, 2],
                        "target": [1, 1, 2, 2],
                        "is_known_positive": [True, False, False, False],
                        "score": [1 for _ in range(4)],
                    }
                ),
                pl.LazyFrame(
                    {
                        "source": [1, 2, 1, 2],
                        "target": [1, 1, 2, 2],
                        "is_known_positive": [False, True, False, False],
                        "score": [2 for _ in range(4)],
                    }
                ),
            ],
            "score_col_name": "score",
        },
        "different_model": {
            "predictions_list": [
                pl.LazyFrame(
                    {
                        "source": [1, 2, 1, 2, 3],  # Extra drug in list compared to base model
                        "target": [1, 1, 2, 2, 1],
                        "is_known_positive": [True, False, False, False],
                        "transformed_score": [3 for _ in range(4)],
                    }
                ),
                pl.LazyFrame(
                    {
                        "source": [1, 2, 1, 2, 3],  # Consistent drug list across the two folds
                        "target": [1, 1, 2, 2, 1],
                        "is_known_positive": [False, True, False, False],
                        "transformed_score": [4 for _ in range(4)],
                    }
                ),
            ],
            "score_col_name": "transformed_score",
        },
    }


@pytest.fixture
def sample_data_inconsistent_folds():
    return {
        "inconsistent_model": {
            "predictions_list": [
                pl.LazyFrame(
                    {
                        "source": [1, 2, 1, 2],
                        "target": [1, 1, 2, 2],
                        "is_known_positive": [True, False, False, False],
                        "score": [1 for _ in range(4)],
                    }
                ),
                pl.LazyFrame(
                    {
                        "source": [1, 2, 1, 2, 3],  # Extra drug in the list compared to the first fold
                        "target": [1, 1, 2, 2, 1],
                        "is_known_positive": [False, True, False, False],
                        "score": [2 for _ in range(4)],
                    }
                ),
            ],
            "score_col_name": "score",
        }
    }


def test_combine_predictions(sample_data_inconsistent_models, sample_data_inconsistent_folds):
    # Given two sets of sample data:
    # 1) Inconsistent drugs list between models but consistent within folds,
    # 2) Inconsistent drugs list between folds
    # When the combine_predictions function is called:
    # a) On sample data 1 with data consistency assertion disabled
    result_harmonization_performed = combine_predictions(
        sample_data_inconsistent_models,
        available_ground_truth_cols=["is_known_positive"],
        perform_multifold=True,
        assert_data_consistency=False,
    )
    # b) On sample data 1 with data consistency assertion enabled
    result_model_inconsistency_exception = combine_predictions(
        sample_data_inconsistent_models,
        available_ground_truth_cols=["is_known_positive"],
        perform_multifold=True,
        assert_data_consistency=True,
    )
    # c) On sample data 1 with multifold disabled
    result_multifold_disabled = combine_predictions(
        sample_data_inconsistent_models,
        available_ground_truth_cols=["is_known_positive"],
        perform_multifold=False,
        assert_data_consistency=False,
    )
    # d) On sample data 2
    result_inconsistent_folds = combine_predictions(
        sample_data_inconsistent_folds,
        available_ground_truth_cols=["is_known_positive"],
        perform_multifold=True,
        assert_data_consistency=False,
    )

    # Then the result is as expected:
    # a) Matrix harmonization is performed
    combined_predictions, predictions_info = result_harmonization_performed
    assert combined_predictions == pl.LazyFrame(
        {
            "source": [1, 2, 1, 2],
            "target": [1, 1, 2, 2],
            "is_known_positive": [True, False, False, False],
        }
    )
    assert predictions_info == {
        "model_names": ["base_model", "different_model"],
        "num_folds": 2,
        "available_ground_truth_cols": ["is_known_positive"],
    }
