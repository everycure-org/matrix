import polars as pl
import pytest
from matrix.pipelines.run_comparison.nodes import combine_matrix_pairs
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
    # When the function is called
    # Then on sample data 1 with data consistency assertion enabled, an exception is raised
    with pytest.raises(ValueError):
        _ = combine_matrix_pairs(
            sample_data_inconsistent_models,
            available_ground_truth_cols=["is_known_positive"],
            perform_multifold=True,
            assert_data_consistency=True,
        )
    # Then on sample data 1 with multifold disabled, matrix pairs are combined for first fold
    combined_matrix_pairs, _ = combine_matrix_pairs(
        sample_data_inconsistent_models,
        available_ground_truth_cols=["is_known_positive"],
        perform_multifold=False,
        assert_data_consistency=False,
    )
    assert list(combined_matrix_pairs.keys()) == ["fold_0"]
    assert_frame_equal(
        combined_matrix_pairs["fold_0"],
        pl.LazyFrame(
            {
                "source": [1, 2, 1, 2],
                "target": [1, 1, 2, 2],
                "is_known_positive": [True, False, False, False],
            }
        ),
        check_row_order=False,
        check_column_order=False,
    )

    # Then on sample data 1 with multifold enables, matrix pairs are combined for all fold
    combined_matrix_pairs, predictions_info = combine_matrix_pairs(
        sample_data_inconsistent_models,
        available_ground_truth_cols=["is_known_positive"],
        perform_multifold=True,
        assert_data_consistency=False,
    )
    assert list(combined_matrix_pairs.keys()) == ["fold_0", "fold_1"]
    assert_frame_equal(
        combined_matrix_pairs["fold_0"],
        pl.LazyFrame(
            {
                "source": [1, 2, 1, 2],
                "target": [1, 1, 2, 2],
                "is_known_positive": [True, False, False, False],
            }
        ),
        check_row_order=False,
        check_column_order=False,
    )
    assert_frame_equal(
        combined_matrix_pairs["fold_1"],
        pl.LazyFrame(
            {
                "source": [1, 2, 1, 2],
                "target": [1, 1, 2, 2],
                "is_known_positive": [False, True, False, False],
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

    # The on sample data 2, an exception is raised due to fold inconsistency
    with pytest.raises(ValueError):
        _ = combine_matrix_pairs(
            sample_data_inconsistent_folds,
            available_ground_truth_cols=["is_known_positive"],
            perform_multifold=True,
            assert_data_consistency=False,
        )
