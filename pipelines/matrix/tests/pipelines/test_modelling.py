import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler, MinMaxScaler

from matrix.pipelines.modelling.nodes import apply_transformers


def test_apply_transformers():
    # Create sample input data with parametrized size
    n_rows = 100000

    data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, n_rows),
            "feature2": np.random.uniform(0, 10, n_rows),
            "feature3": np.random.exponential(2, n_rows),
            "feature4": np.random.choice(["A", "B", "C"], n_rows),
            "feature5": np.random.lognormal(0, 1, n_rows),  # Added: strictly positive data for box-cox
            "non_transform_col": [f"row_{i}" for i in range(n_rows)],
        }
    )

    # Create transformers dictionary with more scalers
    scaler1 = StandardScaler()
    scaler2 = MinMaxScaler()
    scaler3 = RobustScaler()
    scaler4 = QuantileTransformer(output_distribution="normal")
    scaler5 = PowerTransformer(method="box-cox")  # Computationally expensive
    scaler6 = QuantileTransformer(
        n_quantiles=10000, output_distribution="uniform"
    )  # More expensive with many quantiles

    # Fit the transformers on the data and set feature names
    feature1_data = data[["feature1"]]
    feature2_data = data[["feature2"]]
    feature3_data = data[["feature3"]]
    feature5_data = data[["feature5"]]  # Changed: using lognormal data for box-cox

    scaler1.fit(feature1_data)
    scaler2.fit(feature2_data)
    scaler3.fit(feature3_data)
    scaler4.fit(feature3_data)  # Another transformer on feature3
    scaler5.fit(feature5_data)
    scaler6.fit(feature1_data)  # This one can still use feature1 as it doesn't require positive values

    # Set feature names
    for scaler, features in [
        (scaler1, feature1_data),
        (scaler2, feature2_data),
        (scaler3, feature3_data),
        (scaler4, feature3_data),
        (scaler5, feature5_data),  # Changed: using feature5_data instead of feature4_data
        (scaler6, feature1_data),  # Using feature1_data for the dense quantile transformer
    ]:
        scaler.feature_names_in_ = features.columns

    transformers = {
        "standard_scaler": {"transformer": scaler1, "features": ["feature1"]},
        "minmax_scaler": {"transformer": scaler2, "features": ["feature2"]},
        "robust_scaler": {"transformer": scaler3, "features": ["feature3"]},
        "quantile_transformer": {"transformer": scaler4, "features": ["feature3"]},
        "power_transformer": {
            "transformer": scaler5,
            "features": ["feature5"],  # Changed: using feature5 for box-cox
        },
        "quantile_transformer_dense": {"transformer": scaler6, "features": ["feature1"]},
    }

    # Apply transformers
    result = apply_transformers(data, transformers)

    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert "non_transform_col" in result.columns
    assert "feature1" in result.columns
    assert "feature2" in result.columns
    assert result.shape == data.shape
    assert result.columns.tolist() == data.columns.tolist()

    # Check if transformations were applied correctly
    expected_feature1 = scaler6.transform(data[["feature1"]])
    expected_feature2 = scaler2.transform(data[["feature2"]])

    np.testing.assert_array_almost_equal(result["feature1"].values, expected_feature1.flatten(), decimal=2)
    np.testing.assert_array_almost_equal(result["feature2"].values, expected_feature2.flatten(), decimal=2)

    # Check if non-transformed column remains unchanged
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
