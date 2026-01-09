import pandas as pd
import pytest

from core_entities.utils.curation_utils import (
    apply_patch,
)


# ---- Fixtures ----
@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "name": ["a", "b", "c", "d"],
            "value": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def sample_patch_df():
    return pd.DataFrame({"id": ["2", "3"], "name": ["b_updated", "c_updated"], "value": [25, 35]})


# ---- Tests for apply_patch ----
def test_apply_patch_basic(sample_df, sample_patch_df):
    result = apply_patch(sample_df, sample_patch_df, ["name", "value"], "id")

    expected = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "name": ["a", "b_updated", "c_updated", "d"],
            "value": [10, 25, 35, 40],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_apply_patch_with_nonexistent_column():
    df = pd.DataFrame({"id": ["1", "2"], "name": ["a", "b"]})
    patch_df = pd.DataFrame({"id": ["1"], "name": ["a_updated"], "new_col": ["new_value"]})

    with pytest.raises(ValueError, match="Patch column new_col not found in df"):
        apply_patch(df, patch_df, ["name", "new_col"], "id")
