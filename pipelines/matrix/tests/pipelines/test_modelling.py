# Standard library imports
from unittest.mock import Mock

# Third-party imports
import numpy as np
import pandas as pd
import pytest
import pandera
import matplotlib.pyplot as plt

# Machine learning imports
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType,
    IntegerType,
    FloatType,
)

# Local imports
from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import SingleLabelPairGenerator
from matrix.pipelines.modelling.nodes import (
    make_folds,
    create_model_input_nodes,
    prefilter_nodes,
    make_splits,
    attach_embeddings,
    tune_parameters,
)
from matrix.pipelines.modelling.model import ModelWrapper
from matrix.pipelines.modelling.tuning import NopTuner


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


@pytest.fixture(scope="module")
def base_test_data(spark: SparkSession) -> DataFrame:
    # Create test data
    nodes_data = [
        ("node1", ["drug_type1"], None),
        ("node2", ["disease_type1"], None),
        ("node3", ["other_type"], None),
        ("node4", ["drug_type2", "other_type"], None),
    ]
    nodes_schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("all_categories", ArrayType(StringType()), True),
            StructField("topological_embedding", ArrayType(StringType()), True),
        ]
    )
    return spark.createDataFrame(nodes_data, schema=nodes_schema)


@pytest.fixture(scope="module")
def ground_truth_data(spark: SparkSession) -> DataFrame:
    gt_data = [("node1", "node2", 1), ("node3", "node4", 0)]
    gt_schema = StructType(
        [
            StructField("source", StringType(), False),
            StructField("target", StringType(), False),
            StructField("y", IntegerType(), False),
        ]
    )
    return spark.createDataFrame(gt_data, schema=gt_schema)


def test_prefilter_excludes_non_drug_disease_nodes(base_test_data: DataFrame, ground_truth_data: DataFrame) -> None:
    result = prefilter_nodes(
        full_nodes=base_test_data,
        nodes=base_test_data,
        gt=ground_truth_data,
        drug_types=["drug_type1", "drug_type2"],
        disease_types=["disease_type1"],
    )

    result_list = result.collect()
    assert len(result_list) == 3  # Should only include drug/disease nodes
    assert not any(row.id == "node3" for row in result_list)


def test_prefilter_correctly_identifies_drug_nodes(base_test_data: DataFrame, ground_truth_data: DataFrame) -> None:
    result = prefilter_nodes(
        full_nodes=base_test_data,
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


def test_prefilter_correctly_identifies_disease_nodes(base_test_data: DataFrame, ground_truth_data: DataFrame) -> None:
    result = prefilter_nodes(
        full_nodes=base_test_data,
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
    base_test_data: DataFrame, ground_truth_data: DataFrame
) -> None:
    result = prefilter_nodes(
        full_nodes=base_test_data,
        nodes=base_test_data,
        gt=ground_truth_data,
        drug_types=["drug_type1", "drug_type2"],
        disease_types=["disease_type1"],
    )

    result_list = result.collect()

    # Check ground truth positive nodes
    node1 = next(row for row in result_list if row.id == "node1")
    assert node1.is_ground_pos is True

    node2 = next(row for row in result_list if row.id == "node2")
    assert node2.is_ground_pos is True

    node4 = next(row for row in result_list if row.id == "node4")
    assert node4.is_ground_pos is False


@pytest.fixture(scope="module")
def sample_pairs_df(spark: SparkSession) -> DataFrame:
    pairs_data = [("node1", "node2", 1), ("node3", "node4", 0), ("node5", "node6", 1)]
    pairs_schema = StructType(
        [
            StructField("source", StringType(), False),
            StructField("target", StringType(), False),
            StructField("y", IntegerType(), False),
        ]
    )
    return spark.createDataFrame(pairs_data, schema=pairs_schema)


@pytest.fixture(scope="module")
def sample_nodes_df(spark: SparkSession) -> DataFrame:
    nodes_data = [
        ("node1", [0.1, 0.2, 0.3]),
        ("node2", [0.4, 0.5, 0.6]),
        ("node3", [0.7, 0.8, 0.9]),
        ("node4", [1.0, 1.1, 1.2]),
        ("node5", [1.3, 1.4, 1.5]),
        ("node6", [1.6, 1.7, 1.8]),
    ]
    nodes_schema = StructType(
        [StructField("id", StringType(), False), StructField("topological_embedding", ArrayType(FloatType()), False)]
    )
    return spark.createDataFrame(nodes_data, schema=nodes_schema)


def test_attach_embeddings_successful(sample_pairs_df: DataFrame, sample_nodes_df: DataFrame) -> None:
    """Test successful attachment of embeddings to pairs."""
    result = attach_embeddings(sample_pairs_df, sample_nodes_df)

    # Check schema
    assert "source_embedding" in result.columns
    assert "target_embedding" in result.columns
    assert "y" in result.columns

    # Check number of rows preserved
    assert result.count() == sample_pairs_df.count()

    first_row = result.filter(F.col("source") == "node1").first()

    assert np.allclose(first_row["source_embedding"], [0.1, 0.2, 0.3])
    assert np.allclose(first_row["target_embedding"], [0.4, 0.5, 0.6])


def test_attach_embeddings_missing_nodes(
    spark: SparkSession, sample_pairs_df: DataFrame, sample_nodes_df: DataFrame
) -> None:
    """Test handling of pairs with missing nodes."""
    # Add a pair with non-existent nodes
    new_pair = spark.createDataFrame([("nodeX", "nodeY", 1)], schema=sample_pairs_df.schema)
    pairs_with_missing = sample_pairs_df.union(new_pair)

    with pytest.raises(pandera.errors.SchemaError):
        attach_embeddings(pairs_with_missing, sample_nodes_df)


def test_attach_embeddings_empty_pairs(
    spark: SparkSession, sample_pairs_df: DataFrame, sample_nodes_df: DataFrame
) -> None:
    """Test handling of empty pairs dataframe."""
    empty_pairs = spark.createDataFrame([], schema=sample_pairs_df.schema)
    result = attach_embeddings(empty_pairs, sample_nodes_df)
    assert result.count() == 0


def test_attach_embeddings_schema_validation(spark: SparkSession, sample_pairs_df: DataFrame) -> None:
    """Test schema validation with invalid node embeddings."""
    # Create nodes with invalid embedding type (integers instead of floats)
    invalid_nodes_data = [("node1", [1, 2, 3]), ("node2", [4, 5, 6])]
    invalid_nodes_schema = StructType(
        [StructField("id", StringType(), False), StructField("topological_embedding", ArrayType(IntegerType()), False)]
    )
    invalid_nodes_df = spark.createDataFrame(invalid_nodes_data, schema=invalid_nodes_schema)

    with pytest.raises(pandera.errors.SchemaError):
        attach_embeddings(sample_pairs_df, invalid_nodes_df)


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
        }
    )


@pytest.fixture
def simple_splitter():
    """Create a simple K-fold splitter."""
    return KFold(n_splits=2, shuffle=True, random_state=42)


def test_make_splits_basic_functionality(sample_data, simple_splitter):
    """Test basic functionality of make_splits."""
    result = make_splits(data=sample_data, splitter=simple_splitter)

    # Check that all required columns are present
    required_columns = ["source", "source_embedding", "target", "target_embedding", "iteration", "split"]
    assert all(col in result.columns for col in required_columns)

    # Check that we have the same number of rows as input
    assert len(result) == len(sample_data) * 2  # 2 iterations

    # Check that splits are properly labeled
    assert set(result["split"].unique()) == {"TRAIN", "TEST"}

    # Check iterations
    assert set(result["iteration"].unique()) == {0, 1}


def test_make_splits_data_integrity(sample_data, simple_splitter):
    """Test that data values are preserved after splitting."""
    result = make_splits(data=sample_data, splitter=simple_splitter)

    # Check that original values are preserved
    assert set(result["source"].unique()) == set(sample_data["source"].unique())
    assert set(result["target"].unique()) == set(sample_data["target"].unique())


def test_make_splits_schema_validation(sample_data, simple_splitter):
    """Test schema validation with invalid data."""
    # Create invalid data missing required columns
    invalid_data = sample_data.drop(columns=["source_embedding"])

    with pytest.raises(pandera.errors.SchemaError):
        make_splits(data=invalid_data, splitter=simple_splitter)


def test_make_splits_train_test_distribution(sample_data, simple_splitter):
    """Test that each fold has both train and test data."""
    result = make_splits(data=sample_data, splitter=simple_splitter)

    for iteration in result["iteration"].unique():
        iteration_data = result[result["iteration"] == iteration]

        # Check that both train and test splits exist
        assert "TRAIN" in iteration_data["split"].values
        assert "TEST" in iteration_data["split"].values

        # Check that splits are mutually exclusive
        train_indices = set(iteration_data[iteration_data["split"] == "TRAIN"].index)
        test_indices = set(iteration_data[iteration_data["split"] == "TEST"].index)
        assert len(train_indices.intersection(test_indices)) == 0


def test_make_splits_empty_data(simple_splitter):
    """Test behavior with empty input data."""
    empty_data = pd.DataFrame(columns=["source", "source_embedding", "target", "target_embedding", "y"])

    with pytest.raises(ValueError):
        make_splits(data=empty_data, splitter=simple_splitter)


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
            "split": ["TEST", "TRAIN"],
        }
    )


def test_create_model_input_nodes_basic(mock_knowledge_graph, mock_generator, sample_splits):
    result = create_model_input_nodes(graph=mock_knowledge_graph, splits=sample_splits, generator=mock_generator)

    # Check that generator was called with correct arguments
    mock_generator.generate.assert_called_once_with(mock_knowledge_graph, sample_splits)

    # Check that result has correct number of rows (original + generated)
    assert len(result) == len(sample_splits) + len(mock_generator.generate.return_value)

    # Check that generated rows have 'TRAIN' split
    generated_rows = result.iloc[len(sample_splits) :]
    assert all(generated_rows["split"] == "TRAIN")

    # Check that original splits are preserved
    original_rows = result.iloc[: len(sample_splits)]
    assert all(original_rows["split"] == sample_splits["split"])


def test_create_model_input_nodes_empty_splits(mock_knowledge_graph, mock_generator):
    empty_splits = pd.DataFrame(
        {"source": [], "source_embedding": [], "target": [], "target_embedding": [], "iteration": [], "split": []}
    )

    result = create_model_input_nodes(graph=mock_knowledge_graph, splits=empty_splits, generator=mock_generator)

    # Check that only generated data is present
    assert len(result) == len(mock_generator.generate.return_value)
    assert all(result["split"] == "TRAIN")


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
    assert "object" in result
    assert result["object"] == tuner_config["expected_object"]

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
    mock_tuner._estimator = LogisticRegression()
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


def test_make_folds(sample_data, mocker):
    # Mock the settings
    mock_settings = {"DYNAMIC_PIPELINES_MAPPING": {"cross_validation": {"n_splits": 2}}}
    mocker.patch("matrix.settings", mock_settings)

    # Create a simple splitter that splits data into two folds
    class MockSplitter:
        def __init__(self):
            self.n_splits = None

        def split(self, X, y):
            # First fold: first half train, second half test
            fold1 = (list(range(0, 10)), list(range(10, 25)))
            # Second fold: second half train, first half test
            fold2 = (list(range(10, 25)), list(range(0, 10)))
            return [fold1, fold2]

    # Given a splitter with 2 splits
    splitter = MockSplitter()

    # When we make splits
    result = make_folds(sample_data, splitter)

    # Then we get 3 dataframes (2 splits + 1 full dataset)
    print("\n=== Test Data Distribution Analysis ===")

    # The first fold
    fold0 = result[0]
    train_count_0 = len(fold0[fold0["split"] == "TRAIN"])
    test_count_0 = len(fold0[fold0["split"] == "TEST"])
    print("\nFold 0:")
    print(f"Train samples: {train_count_0}")
    print(f"Test samples: {test_count_0}")

    # The second fold
    fold1 = result[1]
    train_count_1 = len(fold1[fold1["split"] == "TRAIN"])
    test_count_1 = len(fold1[fold1["split"] == "TEST"])
    print("\nFold 1:")
    print(f"Train samples: {train_count_1}")
    print(f"Test samples: {test_count_1}")

    # The full dataset
    full_data = result[2]
    print("\nFull Dataset:")
    print(f"Total samples: {len(full_data)}")

    # Test set analysis
    test_indices_fold0 = set(fold0[fold0["split"] == "TEST"].index)
    test_indices_fold1 = set(fold1[fold1["split"] == "TEST"].index)

    print("\n=== Test Set Analysis ===")
    print(f"Test indices in fold 0: {sorted(test_indices_fold0)}")
    print(f"Test indices in fold 1: {sorted(test_indices_fold1)}")

    intersection = test_indices_fold0.intersection(test_indices_fold1)
    print(f"\nOverlap between test sets: {intersection}")

    all_test_indices = test_indices_fold0.union(test_indices_fold1)
    print(f"Combined test indices: {sorted(all_test_indices)}")
    print(f"Total unique test samples: {len(all_test_indices)}")

    # Run the original assertions
    assert len(result) == 3
    assert len(fold0[fold0["split"] == "TRAIN"]) == 10
    assert len(fold0[fold0["split"] == "TEST"]) == 15

    assert len(fold1[fold1["split"] == "TRAIN"]) == 15
    assert len(fold1[fold1["split"] == "TEST"]) == 10

    assert len(full_data) == 25
    assert all(full_data["split"] == "TRAIN")

    # Test set independence assertions
    assert len(test_indices_fold0.intersection(test_indices_fold1)) == 0
    assert len(all_test_indices) == len(sample_data)
    assert all_test_indices == set(range(len(sample_data)))


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
