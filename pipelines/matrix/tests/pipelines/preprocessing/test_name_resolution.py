from unittest.mock import MagicMock, patch

from matrix.pipelines.preprocessing.nodes import (
    add_source_and_target_to_clinical_trials,
    parse_one_name_batch,
    resolve_one_name_batch,
)


def test_add_source_and_target_to_clinical_trials():
    df = pd.DataFrame(
        {
            "clinical_trial_id": ["NCT00118846", "NCT00134030", "NCT00859781"],
            "reason_for_rejection": [None, None, None],
            "drug_name": ["Soy protein", "Ifosfamide", "177Lu-J591"],
            "disease_name": ["Atherosclerosis", "Osteosarcoma", "Prostate Cancer"],
            "significantly_better": [None, 0.0, None],
            "non_significantly_better": [None, 0.0, None],
            "non_significantly_worse": [None, 1.0, None],
            "significantly_worse": [None, 0.0, None],
        }
    )

    mock_resolved_drugs = {
        "Soy protein": {"curie": "RXCUI:196238"},
        "Ifosfamide": {"curie": "CHEBI:5864"},
        "177Lu-J591": {"curie": None},
    }
    mock_resolved_diseases = {
        "Atherosclerosis": {"curie": "MONDO:0005311"},
        "Osteosarcoma": {"curie": "MONDO:0002629"},
        "Prostate Cancer": {"curie": "MONDO:0008315"},
    }

    with patch(
        "matrix.pipelines.preprocessing.nodes.resolve_names", side_effect=[mock_resolved_drugs, mock_resolved_diseases]
    ):
        result_df = add_source_and_target_to_clinical_trials(df, "mock_url", 10)

    expected_df = pd.concat(
        [
            df,
            pd.Series(["RXCUI:196238", "CHEBI:5864", None], name="drug_curie"),
            pd.Series(["MONDO:0005311", "MONDO:0002629", "MONDO:0008315"], name="disease_curie"),
        ],
        axis=1,
    )
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_resolve_names():
    mock_response = {
        "Soy protein": [
            {
                "curie": "RXCUI:196238",
                "label": "soy protein isolate 40 MG Oral Tablet",
                "synonyms": ["soy isoflavone@40 mg@ORAL@TABLET", "soy isoflavone 40 mg ORAL TABLET"],
                "types": ["biolink:Drug", "biolink:ChemicalOrDrugOrTreatment", "biolink:OntologyClass"],
            }
        ],
        "Ifosfamide": [
            {
                "curie": "CHEBI:5864",
                "label": "Ifosfamide",
                "synonyms": ["IFO", "IPP", "IFX", "ipp"],
                "types": ["biolink:SmallMolecule", "biolink:MolecularEntity"],
            }
        ],
        "177Lu-J591": [],
    }
    with patch("matrix.pipelines.preprocessing.nodes.resolve_one_name_batch", return_value=mock_response):
        result = resolve_names(["Soy protein", "Ifosfamide", "177Lu-J591"], ["curie"], "mock_url", 10)
    expected_output = {
        "Soy protein": {"curie": "RXCUI:196238"},
        "Ifosfamide": {"curie": "CHEBI:5864"},
        "177Lu-J591": {"curie": None},
    }
    assert result == expected_output


def test_parse_one_name_batch():
    input_data = {"Soy protein": [{"curie": "RXCUI:196238"}], "Ifosfamide": [{"curie": "CHEBI:5864"}], "177Lu-J591": []}
    expected_output = {
        "Soy protein": {"curie": "RXCUI:196238"},
        "Ifosfamide": {"curie": "CHEBI:5864"},
        "177Lu-J591": {"curie": None},
    }
    result = parse_one_name_batch(input_data, ["curie"])
    assert result == expected_output


@patch("matrix.pipelines.preprocessing.nodes.requests.post")
def test_resolve_one_name_batch_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"resolved_names": ["CURIE:123"]}
    mock_response.elapsed.total_seconds.return_value = 0.5
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    names = ["test_name"]
    url = "http://fakeapi.com/resolve"

    result = resolve_one_name_batch(names, url)
    assert result == {"resolved_names": ["CURIE:123"]}
    mock_post.assert_called_with(
        url,
        json={
            "strings": names,
            "autocomplete": True,
            "highlighting": False,
            "offset": 0,
            "limit": 1,
        },
    )


########## MEDICAL NODES ##########

from unittest.mock import patch

import pandas as pd
import pytest
from matrix.pipelines.preprocessing.nodes import process_medical_nodes, resolve_names


@pytest.fixture
def input_medical_nodes_df():
    data = {
        "ID": [1, 2],
        "name": ["Long COVID", "Long COVID Autonomic Dysfunction"],
        "category": ["Disease", "Disease"],
        "new_id": [None, "EC:1"],
        "description": [None, "Subtype of Long COVID characterised by postural issues"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def expected_output_medical_nodes_df():
    data = {
        "ID": [1, 2],
        "name": ["Long COVID", "Long COVID Autonomic Dysfunction"],
        "category": ["Disease", "Disease"],
        "new_id": [None, "EC:1"],
        "description": [None, "Subtype of Long COVID characterised by postural issues"],
        "curie": ["MONDO:0100233", "MONDO:0100233"],
        "label": ["long COVID-19", "long COVID-19"],
        "types": [
            [
                "biolink:Disease",
                "biolink:DiseaseOrPhenotypicFeature",
                "biolink:Entity",
            ],
            [
                "biolink:Disease",
                "biolink:Entity",
            ],
        ],
        "normalized_curie": ["MONDO:0100233", "EC:1"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_parsed_medical_nodes_names():
    data = {
        "Long COVID": {
            "curie": "MONDO:0100233",
            "label": "long COVID-19",
            "types": [
                "biolink:Disease",
                "biolink:DiseaseOrPhenotypicFeature",
                "biolink:Entity",
            ],
        },
        "Long COVID Autonomic Dysfunction": {
            "curie": "MONDO:0100233",
            "label": "long COVID-19",
            "types": ["biolink:Disease", "biolink:Entity"],
        },
    }
    return data


@pytest.fixture
def mock_resolved_medical_nodes_names():
    data = {
        "Long COVID": [
            {
                "curie": "MONDO:0100233",
                "label": "long COVID-19",
                "highlighting": {},
                "synonyms": [
                    "PASC",
                    "Long COVID",
                    "long COVID-19",
                    "Postacute sequelae of SARS-CoV-2 infection (PASC)",
                ],
                "taxa": [],
                "types": ["biolink:Disease", "biolink:DiseaseOrPhenotypicFeature", "biolink:Entity"],
                "score": 31.311249,
                "clique_identifier_count": 10,
            }
        ],
        "Long COVID Autonomic Dysfunction": [
            {
                "curie": "MONDO:0100233",
                "label": "long COVID-19",
                "highlighting": {},
                "synonyms": [
                    "PASC",
                    "Long COVID",
                    "long COVID",
                    "long COVID-19",
                ],
                "taxa": [],
                "types": ["biolink:Disease", "biolink:Entity"],
                "score": 12.376099,
                "clique_identifier_count": 10,
            }
        ],
    }
    return data


@patch("matrix.pipelines.preprocessing.nodes.resolve_names")
def test_process_medical_nodes(
    mock_resolve_names, input_medical_nodes_df, expected_output_medical_nodes_df, mock_parsed_medical_nodes_names
):
    resolver_url = "http://resolver-url.com"
    batch_size = 50

    mock_resolve_names.return_value = mock_parsed_medical_nodes_names
    output_df = process_medical_nodes(input_medical_nodes_df, resolver_url, batch_size)

    pd.testing.assert_frame_equal(output_df, expected_output_medical_nodes_df)

    mock_resolve_names.assert_called_once_with(
        ["Long COVID", "Long COVID Autonomic Dysfunction"],
        cols_to_get=["curie", "label", "types"],
        url=resolver_url,
        batch_size=batch_size,
    )


def test_parse_one_medical_nodes_name_batch(mock_resolved_medical_nodes_names):
    """Test parsing of API response for resolved names."""
    cols_to_get = ["curie", "label", "types"]

    expected_output = {
        "Long COVID": {
            "curie": "MONDO:0100233",
            "label": "long COVID-19",
            "types": ["biolink:Disease", "biolink:DiseaseOrPhenotypicFeature", "biolink:Entity"],
        },
        "Long COVID Autonomic Dysfunction": {
            "curie": "MONDO:0100233",
            "label": "long COVID-19",
            "types": ["biolink:Disease", "biolink:Entity"],
        },
    }

    result = parse_one_name_batch(mock_resolved_medical_nodes_names, cols_to_get)
    assert result == expected_output
