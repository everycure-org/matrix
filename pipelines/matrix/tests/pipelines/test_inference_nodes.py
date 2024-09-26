import pytest
import pandas as pd
import numpy as np
from matrix.pipelines.inference.nodes import resolve_input_sheet


@pytest.fixture
def sample_data():
    """Dummy data for testing inference type choice."""
    drug_sheet = pd.DataFrame(
        {"curie": ["drug1", "drug2"], "name": ["Drug 1", "Drug 2"]}
    )
    disease_sheet = pd.DataFrame(
        {"curie": ["disease1", "disease2"], "name": ["Disease 1", "Disease 2"]}
    )
    return drug_sheet, disease_sheet


def test_infer_per_drug(sample_data):
    """Test for drug-centric inference request (i.e. where disease_id is nan)."""
    # Given drug, disease, and input sheets required for inference
    drug_sheet, disease_sheet = sample_data
    input_sheet = pd.DataFrame(
        {
            "norm_drug_id": ["drug1"],
            "norm_drug_name": ["Drug 1"],
            "norm_disease_id": [np.nan],
            "norm_disease_name": [np.nan],
            "timestamp": ["2024-01-01 00:00:00"],
        }
    )

    # When resolving the drug and disease sheets to provide input lists of correct length and inference request
    result_type, result_drug_list, result_disease_list = resolve_input_sheet(
        input_sheet, drug_sheet, disease_sheet
    )

    # Then we get correct inference type, drug list and disease lists used as input
    assert result_type["request"] == "Drug-centric predictions"
    assert result_drug_list.equals(
        pd.DataFrame({"curie": ["drug1"], "name": ["Drug 1"]})
    )  # in drug-centric predictions we use one drug only
    assert result_disease_list.equals(
        disease_sheet
    )  # in drug-centric predictions we predict against all diseases


def test_infer_per_disease(sample_data):
    """Test for disease-centric inference request (i.e. where drug_id is nan)."""
    # Given drug, disease, and input sheets required for inference
    drug_sheet, disease_sheet = sample_data
    input_sheet = pd.DataFrame(
        {
            "norm_drug_id": [np.nan],
            "norm_drug_name": [np.nan],
            "norm_disease_id": ["disease1"],
            "norm_disease_name": ["Disease 1"],
            "timestamp": ["2024-01-01 00:00:00"],
        }
    )

    # When resolving the drug and disease sheets to provide input lists of correct length and inference request
    result_type, result_drug_list, result_disease_list = resolve_input_sheet(
        input_sheet, drug_sheet, disease_sheet
    )

    # Then we get correct inference type, drug list and disease lists used as input
    assert result_type["request"] == "Disease-centric predictions"
    assert result_drug_list.equals(
        drug_sheet
    )  # in disease-centric predictions we predict against all drugs
    assert result_disease_list.equals(
        pd.DataFrame({"curie": ["disease1"], "name": ["Disease 1"]})
    )  # in disease-centric predictions we use one disease only


def test_infer_per_pair(sample_data):
    """Test for drug-disease specific inference request (i.e. where drug_id is nan)."""
    # Given drug, disease, and input sheets required for inference
    drug_sheet, disease_sheet = sample_data
    input_sheet = pd.DataFrame(
        {
            "norm_drug_id": ["drug1"],
            "norm_drug_name": ["Drug 1"],
            "norm_disease_id": ["disease1"],
            "norm_disease_name": ["Disease 1"],
            "timestamp": ["2024-01-01 00:00:00"],
        }
    )

    # When resolving the drug and disease sheets to provide input lists of correct length and inference request
    result_type, result_drug_list, result_disease_list = resolve_input_sheet(
        input_sheet, drug_sheet, disease_sheet
    )
    # Then we get correct inference type, drug list and disease lists used as input
    assert result_type["request"] == "Drug-disease specific predictions"
    assert result_drug_list.equals(
        pd.DataFrame({"curie": ["drug1"], "name": ["Drug 1"]})
    )  # one drug id
    assert result_disease_list.equals(
        pd.DataFrame({"curie": ["disease1"], "name": ["Disease 1"]})
    )  # one disease id


def test_empty_drug_and_disease_raises_value_error(sample_data):
    """Test for when no input is provided (error-test)."""
    # Given drug, disease, and empty input sheets required for inference
    drug_sheet, disease_sheet = sample_data
    input_sheet = pd.DataFrame(
        {
            "norm_drug_id": [np.nan],
            "norm_drug_name": [np.nan],
            "norm_disease_id": [np.nan],
            "norm_disease_name": [np.nan],
            "timestamp": ["2024-01-01 00:00:00"],
        }
    )

    # When resolving the drug and disease sheets to provide input lists, we get an error
    with pytest.raises(ValueError, match="Need to specify drug, disease, or both"):
        resolve_input_sheet(input_sheet, drug_sheet, disease_sheet)
