# Standard library imports
from typing import Any, List
from unittest.mock import Mock

import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
import pandas as pd
import pandera

# PySpark imports
import pyspark.sql as ps
import pytest

# Local imports
from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import SingleLabelPairGenerator
from matrix.pipelines.modelling.model import ModelWrapper
from matrix.pipelines.modelling.model_selection import DiseaseAreaSplit
from matrix.pipelines.modelling.nodes import (
    apply_transformers,
    attach_embeddings,
    create_model_input_nodes,
    make_folds,
    prefilter_nodes,
    tune_parameters,
)
from matrix.pipelines.modelling.tuning import NopTuner
from matrix_inject.inject import OBJECT_KW

# Machine learning imports
from sklearn.base import BaseEstimator
from sklearn.impute._base import _BaseImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBClassifier


@pytest.fixture(scope="module")
def base_test_data(spark: ps.SparkSession) -> ps.DataFrame:
    # Create test data
    nodes_data = [
        ("node1", ["drug_type1"], None),
        ("node2", ["disease_type1"], None),
        ("node3", ["other_type"], None),
        ("node4", ["drug_type2", "other_type"], None),
    ]
    nodes_schema = ps.types.StructType(
        [
            ps.types.StructField("id", ps.types.StringType(), False),
            ps.types.StructField("all_categories", ps.types.ArrayType(ps.types.StringType()), True),
            ps.types.StructField("topological_embedding", ps.types.ArrayType(ps.types.StringType()), True),
        ]
    )
    return spark.createDataFrame(nodes_data, schema=nodes_schema)


@pytest.fixture(scope="module")
def ground_truth_data(spark: ps.SparkSession) -> ps.DataFrame:
    gt_data = [("node1", "node2", 1), ("node3", "node4", 0)]
    gt_schema = ps.types.StructType(
        [
            ps.types.StructField("source", ps.types.StringType(), False),
            ps.types.StructField("target", ps.types.StringType(), False),
            ps.types.StructField("y", ps.types.IntegerType(), False),
        ]
    )
    return spark.createDataFrame(gt_data, schema=gt_schema)


@pytest.fixture(scope="module")
def sample_pairs_df(spark: ps.SparkSession) -> ps.DataFrame:
    pairs_data = [("node1", "node2", 1), ("node3", "node4", 0), ("node5", "node6", 1)]
    pairs_schema = ps.types.StructType(
        [
            ps.types.StructField("source", ps.types.StringType(), False),
            ps.types.StructField("target", ps.types.StringType(), False),
            ps.types.StructField("y", ps.types.IntegerType(), False),
        ]
    )
    return spark.createDataFrame(pairs_data, schema=pairs_schema)


@pytest.fixture(scope="module")
def sample_nodes_df(spark: ps.SparkSession) -> ps.DataFrame:
    nodes_data = [
        ("node1", [0.1, 0.2, 0.3]),
        ("node2", [0.4, 0.5, 0.6]),
        ("node3", [0.7, 0.8, 0.9]),
        ("node4", [1.0, 1.1, 1.2]),
        ("node5", [1.3, 1.4, 1.5]),
        ("node6", [1.6, 1.7, 1.8]),
    ]
    nodes_schema = ps.types.StructType(
        [
            ps.types.StructField("id", ps.types.StringType(), False),
            ps.types.StructField("topological_embedding", ps.types.ArrayType(ps.types.FloatType()), False),
        ]
    )
    return spark.createDataFrame(nodes_data, schema=nodes_schema)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "source": ["A", "B", "C", "D", "E", "F"],
            "source_embedding": [np.array([1, 2])] * 6,
            "target": ["X", "Y", "Z", "W", "V", "U"],
            "target_embedding": [np.array([3, 4])] * 6,
            "y": [1, 0, 1, 0, 1, 0],
        },
        index=range(6),  # Add explicit index
    )


@pytest.fixture
def simple_splitter():
    """Create a simple K-fold splitter."""
    return KFold(n_splits=2, shuffle=True, random_state=42)


@pytest.fixture()
def mock_knowledge_graph():
    return Mock(spec=KnowledgeGraph)


@pytest.fixture()
def mock_generator():
    generator = Mock(spec=SingleLabelPairGenerator)
    # Configure mock to return valid data matching schema
    generator.generate.return_value = pd.DataFrame(
        {
            "source": ["drug1", "drug2"],
            "source_embedding": [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
            "target": ["disease1", "disease2"],
            "target_embedding": [np.array([0.5, 0.6]), np.array([0.7, 0.8])],
            "iteration": [0.0, 0.0],
        }
    )
    return generator


@pytest.fixture()
def sample_splits():
    return pd.DataFrame(
        {
            "source": ["drug3", "drug4"],
            "source_embedding": [np.array([0.9, 1.0]), np.array([1.1, 1.2])],
            "target": ["disease3", "disease4"],
            "target_embedding": [np.array([1.3, 1.4]), np.array([1.5, 1.6])],
            "iteration": [1.0, 1.0],
            "fold": [0, 1],
            "split": ["TEST", "TRAIN"],
        }
    )


@pytest.fixture
def tune_data():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame(
        {
            "featureOne_1": np.random.randn(n_samples),
            "featureOne_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "feature_4": np.random.randn(n_samples),
            "feature5_extra": np.random.randn(n_samples),
            "target": np.random.randint(0, 2, n_samples),
            "split": ["TRAIN"] * 80 + ["TEST"] * 20,
        }
    )
    return data


@pytest.fixture
def grid_search_tuner():
    """Create a GridSearchCV tuner."""
    param_grid = {"C": [0.1, 1.0], "max_iter": [100, 200]}
    return GridSearchCV(LogisticRegression(), param_grid, cv=3, scoring="accuracy")


class DummyTransformer(_BaseImputer):
    """
    Test implementation of transformer
    """

    def __init__(self, transformed: List[Any]):
        self.transformed = transformed

    def get_feature_names_out(self, x):
        return x.columns.tolist()

    def transform(self, x):
        return self.transformed


@pytest.fixture
def input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_1": [0, 5, 10],
            "feature_2": [0, 1, 2],
            "non_transform_col": ["row_1", "row_2", "row_3"],
        }
    )


@pytest.mark.parametrize(
    "transformers",
    [
        # Validate single transformer
        {"dummy_transformer_1": {"transformer": DummyTransformer(transformed=[1, 3, 3]), "features": ["feature_1"]}},
        # Validate multiple transformers
        {
            "dummy_transformer_1": {"transformer": DummyTransformer(transformed=[1, 3, 3]), "features": ["feature_1"]},
            "dummy_transformer_2": {"transformer": DummyTransformer(transformed=[2, 6, 6]), "features": ["feature_2"]},
        },
    ],
)
def test_apply_transformers(input_df, transformers):
    # Given input list of transformers

    # When appyling apply transformers
    result = apply_transformers(input_df, transformers)

    # Then output is of correct type, transformers are applied correctly
    # and non_transformer_col is untouched
    assert isinstance(result, pd.DataFrame)
    assert result.shape == input_df.shape
    assert set(result.columns) == set(input_df.columns)

    # NOTEL we currently limiting ourselves to transformers with single input feature
    for _, transformer in transformers.items():
        np.testing.assert_array_equal(result[transformer["features"][0]].values, transformer["transformer"].transformed)

    # # Check if non-transformed column remains unchanged
    assert (result["non_transform_col"] == input_df["non_transform_col"]).all()


def test_prefilter_excludes_non_drug_disease_nodes(
    base_test_data: ps.DataFrame, ground_truth_data: ps.DataFrame
) -> None:
    result = prefilter_nodes(
        nodes=base_test_data,
        gt=ground_truth_data,
        drug_types=["drug_type1", "drug_type2"],
        disease_types=["disease_type1"],
    )

    result_list = result.collect()
    assert len(result_list) == 3  # Should only include drug/disease nodes
    assert not any(row.id == "node3" for row in result_list)


def test_prefilter_correctly_identifies_drug_nodes(
    base_test_data: ps.DataFrame, ground_truth_data: ps.DataFrame
) -> None:
    result = prefilter_nodes(
        nodes=base_test_data,
        gt=ground_truth_data,
        drug_types=["drug_type1", "drug_type2"],
        disease_types=["disease_type1"],
    )

    result_list = result.collect()

    # Check drug nodes
    node1 = next(row for row in result_list if row.id == "node1")
    assert node1.is_drug is True
    assert node1.is_disease is False

    node4 = next(row for row in result_list if row.id == "node4")
    assert node4.is_drug is True
    assert node4.is_disease is False


def test_prefilter_correctly_identifies_disease_nodes(
    base_test_data: ps.DataFrame, ground_truth_data: ps.DataFrame
) -> None:
    result = prefilter_nodes(
        nodes=base_test_data,
        gt=ground_truth_data,
        drug_types=["drug_type1", "drug_type2"],
        disease_types=["disease_type1"],
    )

    result_list = result.collect()

    node2 = next(row for row in result_list if row.id == "node2")
    assert node2.is_drug is False
    assert node2.is_disease is True


def test_prefilter_correctly_identifies_ground_truth_positives(
    base_test_data: ps.DataFrame, ground_truth_data: ps.DataFrame
) -> None:
    result = prefilter_nodes(
        nodes=base_test_data,
        gt=ground_truth_data,
        drug_types=["drug_type1", "drug_type2"],
        disease_types=["disease_type1"],
    )

    result_list = result.collect()

    # Check ground truth positive nodes
    node1 = next(row for row in result_list if row.id == "node1")
    assert node1.in_ground_pos is True

    node2 = next(row for row in result_list if row.id == "node2")
    assert node2.in_ground_pos is True

    node4 = next(row for row in result_list if row.id == "node4")
    assert node4.in_ground_pos is False


def test_attach_embeddings_successful(sample_pairs_df: ps.DataFrame, sample_nodes_df: ps.DataFrame) -> None:
    """Test successful attachment of embeddings to pairs."""
    result = attach_embeddings(sample_pairs_df, sample_nodes_df)

    # Check schema
    assert "source_embedding" in result.columns
    assert "target_embedding" in result.columns
    assert "y" in result.columns

    # Check number of rows preserved
    assert result.count() == sample_pairs_df.count()

    first_row = result.filter(ps.functions.col("source") == "node1").first()

    assert np.allclose(first_row["source_embedding"], [0.1, 0.2, 0.3])
    assert np.allclose(first_row["target_embedding"], [0.4, 0.5, 0.6])


def test_attach_embeddings_missing_nodes(
    spark: ps.SparkSession, sample_pairs_df: ps.DataFrame, sample_nodes_df: ps.DataFrame
) -> None:
    """Test handling of pairs with missing nodes."""
    # Add a pair with non-existent nodes
    new_pair = spark.createDataFrame([("nodeX", "nodeY", 1)], schema=sample_pairs_df.schema)
    pairs_with_missing = sample_pairs_df.union(new_pair)

    with pytest.raises(pandera.errors.SchemaError):
        attach_embeddings(pairs_with_missing, sample_nodes_df)


def test_attach_embeddings_empty_pairs(
    spark: ps.SparkSession, sample_pairs_df: ps.DataFrame, sample_nodes_df: ps.DataFrame
) -> None:
    """Test handling of empty pairs dataframe."""
    empty_pairs = spark.createDataFrame([], schema=sample_pairs_df.schema)
    result = attach_embeddings(empty_pairs, sample_nodes_df)
    assert result.count() == 0


def test_attach_embeddings_schema_validation(spark: ps.SparkSession, sample_pairs_df: ps.DataFrame) -> None:
    """Test schema validation with invalid node embeddings."""
    # Create nodes with invalid embedding type (integers instead of floats)
    invalid_nodes_data = [("node1", [1, 2, 3]), ("node2", [4, 5, 6])]
    invalid_nodes_schema = ps.types.StructType(
        [
            ps.types.StructField("id", ps.types.StringType(), False),
            ps.types.StructField("topological_embedding", ps.types.ArrayType(ps.types.IntegerType()), False),
        ]
    )
    invalid_nodes_df = spark.createDataFrame(invalid_nodes_data, schema=invalid_nodes_schema)

    with pytest.raises(pandera.errors.SchemaError):
        attach_embeddings(sample_pairs_df, invalid_nodes_df)


def test_make_folds_basic_functionality(sample_data, simple_splitter):
    """Test basic functionality of make_folds."""
    result = make_folds(data=sample_data, splitter=simple_splitter)

    # Check that all required columns are present
    required_columns = ["source", "source_embedding", "target", "target_embedding", "split", "fold"]
    assert all(col in result.columns for col in required_columns)

    # Check that we have the same number of rows as input
    assert len(result) == len(sample_data) * 3

    # Check that splits are properly labeled
    assert set(result["split"].unique()) == {"TRAIN", "TEST"}

    # Check that the folds column has the correct range
    assert set(result["fold"].unique()) == {0, 1, 2}


def test_make_folds_data_integrity(sample_data, simple_splitter):
    """Test that data values are preserved after splitting."""
    result = make_folds(data=sample_data, splitter=simple_splitter)

    # Check that original values are preserved
    assert set(result["source"].unique()) == set(sample_data["source"].unique())
    assert set(result["target"].unique()) == set(sample_data["target"].unique())


def test_make_folds_empty_data(simple_splitter):
    """Test behavior with empty input data."""
    empty_data = pd.DataFrame(columns=["source", "source_embedding", "target", "target_embedding", "y"])

    with pytest.raises(ValueError):
        make_folds(data=empty_data, splitter=simple_splitter)


def test_create_model_input_nodes_basic(mock_knowledge_graph, mock_generator, sample_data, simple_splitter):
    # Get number of folds (test/train splits plus full training set)
    n_folds = simple_splitter.get_n_splits() + 1

    # Given the output of make_folds
    mock_splits = make_folds(data=sample_data, splitter=simple_splitter)

    # When creating model input nodes
    result = create_model_input_nodes(graph=mock_knowledge_graph, splits=mock_splits, generator=mock_generator)

    # Check that generator was called the correct number of times (once per fold)
    assert mock_generator.generate.call_count == n_folds

    # Check that result has correct number of rows (original + generated * number of folds)
    assert len(result) == len(mock_splits) + len(mock_generator.generate.return_value) * n_folds

    # Check that generated rows have 'TRAIN' split
    generated_rows = result.iloc[len(mock_splits) :]
    assert all(generated_rows["split"] == "TRAIN")

    # Check that original splits are preserved
    original_rows = result.iloc[: len(mock_splits)]
    assert all(original_rows["split"] == mock_splits["split"])


def test_create_model_input_nodes_generator_empty(mock_knowledge_graph, sample_splits):
    # Test with generator that returns empty DataFrame
    empty_generator = Mock(spec=SingleLabelPairGenerator)
    empty_generator.generate.return_value = pd.DataFrame(
        {"source": [], "source_embedding": [], "target": [], "target_embedding": [], "iteration": []}
    )

    result = create_model_input_nodes(graph=mock_knowledge_graph, splits=sample_splits, generator=empty_generator)

    # Check that only original splits are present
    assert len(result) == len(sample_splits)
    assert all(result["split"] == sample_splits["split"])


@pytest.mark.parametrize(
    "tuner_config",
    [
        {
            "tuner_factory": lambda: NopTuner(
                XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            ),
            "expected_object": "xgboost.sklearn.XGBClassifier",
            "expected_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
            "plot_required": False,
        },
        {
            "tuner_factory": lambda: GridSearchCV(
                LogisticRegression(), {"C": [0.1, 1.0], "max_iter": [100, 200]}, cv=3, scoring="accuracy"
            ),
            "expected_object": "sklearn.linear_model._logistic.LogisticRegression",
            "expected_params": {"C", "max_iter"},
            "plot_required": True,
        },
    ],
)
def test_tune_parameters(tune_data: pd.DataFrame, tuner_config: dict):
    """Test parameter tuning functionality with different tuners."""
    tuner = tuner_config["tuner_factory"]()

    result, plot = tune_parameters(
        data=tune_data,
        tuner=tuner,
        features=["featureOne_1", "featureOne_2"],
        target_col_name="target",
    )

    # Check return value structure
    assert isinstance(result, dict)
    assert OBJECT_KW in result
    assert result[OBJECT_KW] == tuner_config["expected_object"]

    # Check parameters
    if isinstance(tuner_config["expected_params"], dict):
        # For exact parameter matching (NopTuner case)
        for param, value in tuner_config["expected_params"].items():
            assert result[param] == value
    else:
        # For parameter presence checking (GridSearchCV case)
        for param in tuner_config["expected_params"]:
            assert param in result

    # Check plot expectations
    if tuner_config["plot_required"]:
        assert isinstance(plot, plt.Figure)
    else:
        assert plot is None or isinstance(plot, plt.Figure)


def test_tune_parameters_regex_features(tune_data: pd.DataFrame, grid_search_tuner: GridSearchCV) -> None:
    """Test regex feature selection."""
    result, _ = tune_parameters(
        data=tune_data, tuner=grid_search_tuner, features=["featureOne.*"], target_col_name="target"
    )

    mask = tune_data["split"] == "TRAIN"
    train_data = tune_data[mask]
    feature_cols = train_data.filter(regex="featureOne.*").columns
    # Should have used all three features matching 'featureOne.*'
    assert len(feature_cols) == 2


def test_tune_parameters_regex_features_other_convention(
    tune_data: pd.DataFrame, grid_search_tuner: GridSearchCV
) -> None:
    """Test regex feature selection."""
    result, _ = tune_parameters(
        data=tune_data, tuner=grid_search_tuner, features=["feature_+"], target_col_name="target"
    )

    mask = tune_data["split"] == "TRAIN"
    train_data = tune_data[mask]
    feature_cols = train_data.filter(regex="feature_+").columns
    # Should have used all three features matching 'feature_+'
    assert len(feature_cols) == 2


def test_tune_parameters_train_test_split(tune_data: pd.DataFrame, grid_search_tuner: GridSearchCV) -> None:
    """Test proper handling of train/test splits."""
    mock_tuner = Mock()
    mock_tuner.estimator = LogisticRegression()
    mock_tuner.best_params_ = {"C": 1.0, "max_iter": 100}

    result, _ = tune_parameters(
        data=tune_data, tuner=mock_tuner, features=["featureOne_1", "featureOne_2"], target_col_name="target"
    )

    # Verify only training data was used
    train_mask = tune_data["split"] == "TRAIN"
    expected_train_samples = len(tune_data[train_mask])

    # Check that fit was called with correct number of samples
    args, _ = mock_tuner.fit.call_args
    assert len(args[0]) == expected_train_samples


def test_tune_parameters_invalid_features(tune_data: pd.DataFrame, grid_search_tuner: GridSearchCV) -> None:
    """Test handling of invalid feature names."""
    with pytest.raises(ValueError):
        tune_parameters(
            data=tune_data, tuner=grid_search_tuner, features=["nonexistent_feature"], target_col_name="target"
        )


def test_tune_parameters_convergence_plot(tune_data: pd.DataFrame) -> None:
    """Test convergence plot generation with custom tuner."""

    class CustomTuner:
        def __init__(self):
            self.estimator = LogisticRegression()
            self.best_params_ = {"C": 1.0}
            self.convergence_plot = plt.figure()

        def fit(self, X, y):
            return self

    custom_tuner = CustomTuner()
    _, plot = tune_parameters(data=tune_data, tuner=custom_tuner, features=["featureOne_1"], target_col_name="target")

    assert isinstance(plot, plt.Figure)
    assert plot == custom_tuner.convergence_plot


def test_model_wrapper():
    class MyEstimator(BaseEstimator):
        def __init__(self, proba):
            self.proba = proba
            super().__init__()

        def predict_proba(self, X):
            return self.proba

    my_estimators = [
        MyEstimator(proba=[1, 2, 3]),
        MyEstimator(proba=[2, 3, 5]),
    ]

    # given an instance of a model wrapper with mean
    model_mean = ModelWrapper(estimators=my_estimators, agg_func=np.mean)
    # when invoking the predict_proba
    proba_mean = model_mean.predict_proba([])
    # then median computed correctly
    assert np.all(proba_mean == [1.5, 2.5, 4.0])


@pytest.fixture
def disease_list():
    """Create a sample disease list for testing."""
    return pd.DataFrame(
        {
            "id": ["X", "Y", "Z", "W", "V", "U"],  # Match the target values in sample_data
            "harrisons_view": ["type1", "type2", "type3", "type1", "type2", "type3"],
        },
        index=range(6),  # Add explicit index to match sample_data
    )


def test_make_folds_with_disease_area_split(sample_data, disease_list):
    """Test make_folds with DiseaseAreaSplit."""
    disease_area_splitter = DiseaseAreaSplit(
        n_splits=3, disease_grouping_type="harrisons_view", holdout_disease_types=["type1", "type2", "type3"]
    )

    # Ensure disease_list has the correct format
    assert set(disease_list["id"].tolist()) == set(sample_data["target"].tolist())

    result = make_folds(data=sample_data, splitter=disease_area_splitter, disease_list=disease_list)

    # Check that all required columns are present
    required_columns = ["source", "source_embedding", "target", "target_embedding", "split", "fold"]
    assert all(col in result.columns for col in required_columns)

    # Check that we have the same number of rows as input
    assert len(result) == len(sample_data) * 4  # 3 folds + 1 full training set

    # Check that splits are properly labeled
    assert set(result["split"].unique()) == {"TRAIN", "TEST"}

    # Check that the folds column has the correct range
    assert set(result["fold"].unique()) == {0, 1, 2, 3}
