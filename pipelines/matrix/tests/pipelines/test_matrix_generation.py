from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql as ps
import pytest
from matrix.pipelines.matrix_generation.nodes import (
    generate_pairs,
    generate_reports,
    make_predictions_and_sort,
    package_model_with_transformers,
)
from matrix.pipelines.modelling.model import ModelWrapper
from matrix.pipelines.modelling.transformers import FlatArrayTransformer


@pytest.fixture
def sample_drugs(spark):
    """Fixture that provides sample drugs data for testing."""
    return spark.createDataFrame(
        pd.DataFrame(
            {
                "ec_id": ["ec_drug_1", "ec_drug_2"],
                "id": ["drug_1", "drug_2"],
                "name": ["Drug 1", "Drug 2"],
                "description": ["Description 1", "Description 2"],
            }
        )
    )


@pytest.fixture
def sample_diseases(spark):
    """Fixture that provides sample diseases data for testing."""
    return spark.createDataFrame(
        pd.DataFrame(
            {
                "id": ["disease_1", "disease_2"],
                "name": ["Disease 1", "Disease 2"],
                "description": ["Description A", "Description B"],
            }
        )
    )


@pytest.fixture
def sample_known_pairs(spark):
    """Fixture that provides sample known pairs data for testing."""
    return spark.createDataFrame(
        pd.DataFrame(
            {
                "source": ["drug_1", "drug_2", "drug_3"],
                "source_ec_id": ["ec_drug_1", "ec_drug_2", "ec_drug_3"],
                "target": ["disease_1", "disease_2", "disease_3"],
                "split": ["TRAIN", "TEST", "TRAIN"],
                "y": [1, 0, 1],
            }
        )
    )


@pytest.fixture
def sample_clinical_trials(spark):
    """Fixture that provides sample clinical trials data for testing."""
    return spark.createDataFrame(
        pd.DataFrame(
            {
                "subject": ["drug_1"],
                "subject_ec_id": ["ec_drug_1"],
                "object": ["disease_2"],
                "significantly_better": [1],
                "non_significantly_better": [0],
                "non_significantly_worse": [0],
                "significantly_worse": [0],
            }
        )
    )


@pytest.fixture
def sample_off_label(spark):
    """Fixture that provides sample off label data for testing."""
    return spark.createDataFrame(
        pd.DataFrame(
            {
                "subject": ["drug_1"],
                "subject_ec_id": ["ec_drug_1"],
                "object": ["disease_2"],
                "off_label": [1],
            }
        )
    )


@pytest.fixture
def sample_orchard(spark):
    """Fixture that provides sample orchard data for testing."""
    return spark.createDataFrame(
        pd.DataFrame(
            {
                "subject": ["drug_1"],
                "object": ["disease_2"],
                "high_evidence_matrix": [1],
                "high_evidence_crowdsourced": [0],
                "mid_evidence_matrix": [0],
                "mid_evidence_crowdsourced": [0],
                "archive_biomedical_review": [0],
            }
        )
    )


@pytest.fixture
def sample_node_embeddings(spark):
    """Fixture that provides sample node embeddings for testing."""
    return spark.createDataFrame(
        pd.DataFrame(
            {
                "id": ["drug_1", "drug_2", "disease_1", "disease_2"],
                "is_drug": [True, True, False, False],
                "is_disease": [False, False, True, True],
                "topological_embedding": [np.ones(3).tolist() * n for n in range(4)],
            }
        )
    )


@pytest.fixture
def sample_matrix_data(spark):
    return spark.createDataFrame(
        pd.DataFrame(
            [
                {"ec_drug_id": "ec_drug_1", "source": "drug_1", "target": "disease_1"},
                {"ec_drug_id": "ec_drug_2", "source": "drug_2", "target": "disease_1"},
                {"ec_drug_id": "ec_drug_1", "source": "drug_1", "target": "disease_2"},
                {"ec_drug_id": "ec_drug_2", "source": "drug_2", "target": "disease_2"},
            ]
        )
    )


@pytest.fixture
def transformers():
    return {
        "flat_source_array": {
            "transformer": FlatArrayTransformer(prefix="source_"),
            "features": ["source_embedding"],
        },
        "flat_target_array": {
            "transformer": FlatArrayTransformer(prefix="target_"),
            "features": ["target_embedding"],
        },
    }


@pytest.fixture
def mock_model():  # Note: gives correct shaped output only for batches of size 2
    model = Mock()
    model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1], [0.1, 0.7, 0.2]])
    return model


@pytest.fixture
def mock_model_2():
    model = Mock()
    model.predict_proba = lambda x: np.array([[1, 0, 0]] * len(x))
    return model


def test_generate_pairs(
    sample_drugs,
    sample_diseases,
    sample_node_embeddings,
    sample_known_pairs,
    sample_clinical_trials,
    sample_off_label,
    sample_orchard,
):
    """Test the generate_pairs function."""
    # Given drug list, disease list and ground truth pairs
    # When generating the matrix dataset
    result = generate_pairs(
        drugs=sample_drugs,
        diseases=sample_diseases,
        node_embeddings=sample_node_embeddings,
        known_pairs=sample_known_pairs,
        clinical_trials=sample_clinical_trials,
        off_label=sample_off_label,
        orchard=sample_orchard,
    ).toPandas()

    # Then the output is of the correct format and shape
    assert isinstance(result, pd.DataFrame)
    assert {"source", "target"}.issubset(set(result.columns))
    assert len(result) == 3  # 2 drugs * 2 diseases - 1 training pair
    # Doesn't contain training pairs
    assert not result.apply(
        lambda row: (row["source"], row["source_ec_id"], row["target"]) in [("drug_1", "ec_drug_1", "disease_1")],
        axis=1,
    ).any()

    # Boolean flag columns are present
    def check_col(col_name):
        return (col_name in result.columns) and result[col_name].dtype == bool

    assert all(
        check_col(col)
        for col in [
            "is_known_positive",
            "is_known_negative",
            "trial_sig_better",
            "trial_non_sig_better",
            "trial_sig_worse",
            "trial_non_sig_worse",
        ]
    )
    assert all(
        check_col(col)
        for col in [
            "is_known_positive",
            "is_known_negative",
            "trial_sig_better",
            "trial_non_sig_better",
            "trial_sig_worse",
            "trial_non_sig_worse",
        ]
    )
    # Flag columns set correctly
    assert all(
        [
            sum(result[col_name]) == 1
            for col_name in (
                "is_known_negative",
                "trial_sig_better",
                "high_evidence_matrix",
            )
        ]
    )
    assert all(
        [
            sum(result[col_name]) == 0
            for col_name in (
                "is_known_positive",
                "trial_non_sig_better",
                "trial_sig_worse",
                "trial_non_sig_worse",
                "mid_evidence_matrix",
                "mid_evidence_crowdsourced",
                "archive_biomedical_review",
            )
        ]
    )


def test_generate_pairs_without_ec_id(
    sample_drugs: ps.DataFrame,
    sample_diseases: ps.DataFrame,
    sample_node_embeddings: ps.DataFrame,
    sample_known_pairs: ps.DataFrame,
    sample_clinical_trials: ps.DataFrame,
    sample_off_label: ps.DataFrame,
    sample_orchard: ps.DataFrame,
):
    """Test the generate_pairs function with ."""
    # Given drug list, disease list and ground truth pairs
    # When generating the matrix dataset
    result = generate_pairs(
        drugs=sample_drugs.drop("ec_id"),
        diseases=sample_diseases,
        node_embeddings=sample_node_embeddings,
        known_pairs=sample_known_pairs.drop("source_ec_id"),
        clinical_trials=sample_clinical_trials.drop("source_ec_id"),
        off_label=sample_off_label.drop("source_ec_id"),
        orchard=sample_orchard,
    ).toPandas()

    # Then the output is of the correct format and shape
    assert isinstance(result, pd.DataFrame)
    assert {"source", "target"}.issubset(set(result.columns))
    assert len(result) == 3  # 2 drugs * 2 diseases - 1 training pair
    # Doesn't contain training pairs
    assert not result.apply(lambda row: (row["source"], row["target"]) in [("drug_1", "disease_1")], axis=1).any()

    # Boolean flag columns are present
    def check_col(col_name):
        return (col_name in result.columns) and result[col_name].dtype == bool

    assert all(
        check_col(col)
        for col in [
            "is_known_positive",
            "is_known_negative",
            "trial_sig_better",
            "trial_non_sig_better",
            "trial_sig_worse",
            "trial_non_sig_worse",
        ]
    )
    assert all(
        check_col(col)
        for col in [
            "is_known_positive",
            "is_known_negative",
            "trial_sig_better",
            "trial_non_sig_better",
            "trial_sig_worse",
            "trial_non_sig_worse",
        ]
    )
    # Flag columns set correctly
    assert all(
        [
            sum(result[col_name]) == 1
            for col_name in (
                "is_known_negative",
                "trial_sig_better",
                "high_evidence_matrix",
            )
        ]
    )
    assert all(
        [
            sum(result[col_name]) == 0
            for col_name in (
                "is_known_positive",
                "trial_non_sig_better",
                "trial_sig_worse",
                "trial_non_sig_worse",
                "mid_evidence_matrix",
                "mid_evidence_crowdsourced",
                "archive_biomedical_review",
            )
        ]
    )


def test_make_predictions_and_sort(
    spark,
    sample_node_embeddings,
    sample_matrix_data,
    transformers,
    mock_model,
    mock_model_2,
):
    # Test uses 2 models to match the settings (xg_ensemble, xg_synth)
    base_wrapper_1 = ModelWrapper([mock_model], np.mean)
    base_wrapper_2 = ModelWrapper([mock_model_2], np.mean)

    model_wrapper_1 = package_model_with_transformers(
        transformers,
        base_wrapper_1,
        ["source_+", "target_+"],
    )
    model_wrapper_2 = package_model_with_transformers(
        transformers,
        base_wrapper_2,
        ["source_+", "target_+"],
    )

    ensemble_model = ModelWrapper([model_wrapper_1, model_wrapper_2], np.mean)

    result = make_predictions_and_sort(
        sample_node_embeddings,  # node_embeddings
        sample_matrix_data,  # pairs
        "treat score",
        "not treat score",
        "unknown score",
        ensemble_model,
    )

    assert isinstance(result, ps.DataFrame)
    result_pandas = result.toPandas()
    assert "treat score" in result_pandas.columns
    assert len(result_pandas) == 4
    assert result_pandas["treat score"].is_monotonic_decreasing


@pytest.fixture
def mock_reporting_plot_strategies():
    names = ["strategy_1", "strategy_2"]
    generators = {name: Mock() for name in names}
    for name, generator in generators.items():
        generator.name = name
        generator.generate.return_value = plt.Figure()
    return generators


@pytest.fixture
def mock_reporting_table_strategies():
    names = ["strategy_1", "strategy_2"]
    generators = {name: Mock() for name in names}
    for name, generator in generators.items():
        generator.name = name
        generator.generate.return_value = pd.DataFrame()
    return generators


def test_generate_report_plot(
    sample_matrix_data,
    mock_reporting_plot_strategies,
):
    # Given a list of reporting plot generator and a matrix dataframe
    # When generating the report plot
    reports_dict = generate_reports(sample_matrix_data, mock_reporting_plot_strategies)
    # Then:
    # The keys of the dictionary are the names of the reporting plot generator and have the correct suffix
    assert list(reports_dict.keys()) == ["strategy_1", "strategy_2"]
    # The values are the reporting plot generator's generate method's output
    assert all(isinstance(value, plt.Figure) for value in reports_dict.values())


def test_generate_report_table(
    sample_matrix_data,
    mock_reporting_table_strategies,
):
    # Given a list of reporting table generator and a matrix dataframe
    # When generating the report table
    reports_dict = generate_reports(sample_matrix_data, mock_reporting_table_strategies)
    # Then:
    # The keys of the dictionary are the names of the reporting table generator and have the correct suffix
    assert list(reports_dict.keys()) == ["strategy_1", "strategy_2"]
    # The values are the reporting table generator's generate method's output
    assert all(isinstance(value, pd.DataFrame) for value in reports_dict.values())
