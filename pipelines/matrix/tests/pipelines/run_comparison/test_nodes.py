import polars as pl
import pytest
from matrix.pipelines.run_comparison.nodes import combine_predictions
from polars.testing import assert_frame_equal


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
                        "is_known_positive": [True, False, False, False, False],  # Same as fold 0 of base model
                        "transformed_score": [3 for _ in range(5)],
                    }
                ),
                pl.LazyFrame(
                    {
                        "source": [1, 2, 1, 2, 3],  # Consistent drug list across the two folds
                        "target": [1, 1, 2, 2, 1],
                        "is_known_positive": [False, True, False, False, False],  # Same as fold 1 of base model
                        "transformed_score": [4 for _ in range(5)],
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
                        "is_known_positive": [False, True, False, False, False],
                        "score": [2 for _ in range(5)],
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
    # When the combine_predictions function is called
    # Then on sample data 1 with data consistency assertion disabled, matrix harmonization is performed
    combined_predictions, predictions_info = combine_predictions(
        sample_data_inconsistent_models,
        available_ground_truth_cols=["is_known_positive"],
        perform_multifold=True,
        assert_data_consistency=False,
    )
    assert_frame_equal(
        combined_predictions,
        pl.LazyFrame(
            {
                "source": [1, 2, 1, 2],
                "target": [1, 1, 2, 2],
                "is_known_positive_fold_0": [True, False, False, False],
                "is_known_positive_fold_1": [False, True, False, False],
                "score_base_model_fold_0": [1 for _ in range(4)],
                "score_base_model_fold_1": [2 for _ in range(4)],
                "score_different_model_fold_0": [3 for _ in range(4)],
                "score_different_model_fold_1": [4 for _ in range(4)],
            }
        ),
        check_row_order=False,
        check_column_order=False,
    )
    assert predictions_info == {
        "model_names": ["base_model", "different_model"],
        "num_folds": 2,
        "available_ground_truth_cols": ["is_known_positive"],
    }
    # Then on sample data 1 with data consistency assertion enabled, an exception is raised
    with pytest.raises(ValueError):
        _ = combine_predictions(
            sample_data_inconsistent_models,
            available_ground_truth_cols=["is_known_positive"],
            perform_multifold=True,
            assert_data_consistency=True,
        )
    # Then on sample data 1 with multifold disabled, the first fold is used
    combined_predictions, _ = combine_predictions(
        sample_data_inconsistent_models,
        available_ground_truth_cols=["is_known_positive"],
        perform_multifold=False,
        assert_data_consistency=False,
    )
    assert_frame_equal(
        combined_predictions,
        pl.LazyFrame(
            {
                "source": [1, 2, 1, 2],
                "target": [1, 1, 2, 2],
                "is_known_positive_fold_0": [True, False, False, False],
                "score_base_model_fold_0": [1 for _ in range(4)],
                "score_different_model_fold_0": [3 for _ in range(4)],
            }
        ),
        check_row_order=False,
        check_column_order=False,
    )

    # d) On sample data 2, an exception is raised due to fold inconsistency
    with pytest.raises(ValueError):
        _ = combine_predictions(
            sample_data_inconsistent_folds,
            available_ground_truth_cols=["is_known_positive"],
            perform_multifold=True,
            assert_data_consistency=False,
        )
