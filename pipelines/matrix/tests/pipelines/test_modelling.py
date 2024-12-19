import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from matrix.pipelines.modelling.nodes import apply_transformers


def test_apply_single_transformer():
    data = pd.DataFrame(
        {
            "feature1": [0, 5, 10],  # Will be scaled to [0, 0.5, 1]
            "non_transform_col": ["row_1", "row_2", "row_3"],
        }
    )

    scaler = MinMaxScaler()
    scaler.fit(data[["feature1"]])
    scaler.feature_names_in_ = ["feature1"]

    transformers = {
        "minmax_scaler": {"transformer": scaler, "features": ["feature1"]},
    }

    result = apply_transformers(data, transformers)

    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert result.shape == data.shape
    assert set(result.columns) == set(data.columns)

    expected_feature1 = [0.0, 0.5, 1.0]
    np.testing.assert_array_almost_equal(result["feature1"].values, expected_feature1, decimal=3)

    # Check if non-transformed column remains unchanged
    assert (result["non_transform_col"] == data["non_transform_col"]).all()


def test_apply_multiple_transformers():
    # Test with two simple transformers
    data = pd.DataFrame(
        {
            "feature1": [0, 5, 10],  # For MinMaxScaler
            "feature2": [0, 5, 10],  # For StandardScaler
            "non_transform_col": ["row_1", "row_2", "row_3"],
        }
    )

    # Create and fit transformers
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    minmax_scaler.fit(data[["feature1"]])
    standard_scaler.fit(data[["feature2"]])

    minmax_scaler.feature_names_in_ = ["feature1"]
    standard_scaler.feature_names_in_ = ["feature2"]

    transformers = {
        "minmax_scaler": {"transformer": minmax_scaler, "features": ["feature1"]},
        "standard_scaler": {"transformer": standard_scaler, "features": ["feature2"]},
    }

    # Apply transformers
    result = apply_transformers(data, transformers)

    # Only verify that both transformations were applied
    assert isinstance(result, pd.DataFrame)
    assert result.shape == data.shape
    assert set(result.columns) == set(data.columns)
    assert not (result["feature1"] == data["feature1"]).all()
    assert not (result["feature2"] == data["feature2"]).all()
    assert (result["non_transform_col"] == data["non_transform_col"]).all()


def test_apply_transformers_empty_transformers():
    # Test with empty transformers dictionary
    data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

    result = apply_transformers(data, {})

    assert isinstance(result, pd.DataFrame)
    assert result.equals(data)


def test_apply_transformers_invalid_features():
    # Test with invalid feature names
    data = pd.DataFrame({"feature1": [1, 2, 3]})

    scaler = StandardScaler()
    scaler.fit(data[["feature1"]])

    transformers = {"scaler1": {"transformer": scaler, "features": ["non_existent_feature"]}}

    with pytest.raises(KeyError):
        apply_transformers(data, transformers)
