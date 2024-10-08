import unittest
from unittest.mock import patch, MagicMock
import requests

from matrix.pipelines.preprocessing.nodes import resolve, normalize


class TestResolveFunction(unittest.TestCase):
    @patch("requests.get")
    def test_resolve_success(self, mock_get):
        # Mimic the API with actual example from the synonymizer
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "CHEBI:100147": {
                "preferred_curie": "CHEMBL.COMPOUND:CHEMBL5",
                "preferred_name": "NALIDIXIC ACID",
                "preferred_label": "biolink:Drug",
            }
        }
        mock_get.return_value = mock_response

        # When calling the synonymizer, yield the response
        result = resolve(name="CHEBI:100147", endpoint="http://dummy-endpoint")

        # Assert the correct curie is returned
        self.assertEqual(result, "CHEMBL.COMPOUND:CHEMBL5")

    @patch("requests.get")
    def test_resolve_success_name(self, mock_get):
        # Mimic the API with actual example from the synonymizer
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "CHEBI:100147": {
                "preferred_curie": "CHEMBL.COMPOUND:CHEMBL5",
                "preferred_name": "NALIDIXIC ACID",
                "preferred_label": "biolink:Drug",
            }
        }
        mock_get.return_value = mock_response

        # When calling the synonymizer, yield the response
        result = resolve(
            name="CHEBI:100147",
            endpoint="http://dummy-endpoint",
            att_to_get="preferred_name",
        )

        # Assert the correct curie is returned
        self.assertEqual(result, "NALIDIXIC ACID")

    @patch("requests.get")
    def test_resolve_no_attribute_found(self, mock_get):
        # Mimic the API with actual example from the synonymizer
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "CHEBI:100147": {
                "preferred_curie": "CHEMBL.COMPOUND:CHEMBL5",
                "preferred_name": "NALIDIXIC ACID",
                "preferred_label": "biolink:Drug",
            }
        }
        mock_get.return_value = mock_response

        # When calling the synonymizer, yield the response
        result = resolve(name="CHEBI:12345", endpoint="http://dummy-endpoint")

        # Assert that None is returned when the curie is missing
        self.assertIsNone(result)


class TestNormalizeFunction(unittest.TestCase):
    @patch("requests.get")
    def test_curie_normalize_success(self, mock_get):
        # Mimic the API with actual example from the synonymizer
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "MONDO:0005260": {
                "id": {
                    "identifier": "MONDO:0005258",
                    "name": "autism spectrum disorder",
                    "category": "biolink:Disease",
                    "SRI_normalizer_curie": "MONDO:0005258",
                    "SRI_normalizer_name": "autism spectrum disorder",
                    "SRI_normalizer_category": "biolink:Disease",
                }
            }
        }
        mock_get.return_value = mock_response

        # When calling the normalizer, yield the response
        result = normalize(curie="MONDO:0005260", endpoint="http://dummy-endpoint")

        # Assert the correct curie is returned
        self.assertEqual(result, "MONDO:0005258")

    @patch("requests.get")
    def test_name_normalize_success(self, mock_get):
        # Mimic the API with actual example from the synonymizer
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "autism": {
                "id": {
                    "identifier": "MONDO:0005258",
                    "name": "autism spectrum disorder",
                    "category": "biolink:Disease",
                    "SRI_normalizer_curie": "MONDO:0005258",
                    "SRI_normalizer_name": "autism spectrum disorder",
                    "SRI_normalizer_category": "biolink:Disease",
                }
            }
        }
        mock_get.return_value = mock_response

        # When calling the normalizer, yield the response
        result = normalize(curie="autism", endpoint="http://dummy-endpoint")

        # Assert the correct curie is returned
        self.assertEqual(result, "MONDO:0005258")

    @patch("requests.get")
    def test_curie_normalize_success_name(self, mock_get):
        # Mimic the API with actual example from the synonymizer
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "MONDO:0005260": {
                "id": {
                    "identifier": "MONDO:0005258",
                    "name": "autism spectrum disorder",
                    "category": "biolink:Disease",
                    "SRI_normalizer_curie": "MONDO:0005258",
                    "SRI_normalizer_name": "autism spectrum disorder",
                    "SRI_normalizer_category": "biolink:Disease",
                }
            }
        }
        mock_get.return_value = mock_response

        # When calling the normalizer, yield the response
        result = normalize(
            curie="MONDO:0005260", endpoint="http://dummy-endpoint", att_to_get="name"
        )

        # Assert the correct curie is returned
        self.assertEqual(result, "autism spectrum disorder")

    @patch("requests.get")
    def test_name_normalize_success_name(self, mock_get):
        # Mimic the API with actual example from the synonymizer
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "autism": {
                "id": {
                    "identifier": "MONDO:0005258",
                    "name": "autism spectrum disorder",
                    "category": "biolink:Disease",
                    "SRI_normalizer_curie": "MONDO:0005258",
                    "SRI_normalizer_name": "autism spectrum disorder",
                    "SRI_normalizer_category": "biolink:Disease",
                }
            }
        }
        mock_get.return_value = mock_response

        # When calling the normalizer, yield the response
        result = normalize(
            curie="autism", endpoint="http://dummy-endpoint", att_to_get="name"
        )

        # Assert the correct curie is returned
        self.assertEqual(result, "autism spectrum disorder")

    @patch("requests.get")
    def test_normalize_no_attribute_found(self, mock_get):
        # Mimic the API with actual example from the synonymizer
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "MONDO:0005260": {
                "id": {
                    "identifier": "MONDO:0005258",
                    "name": "autism spectrum disorder",
                    "category": "biolink:Disease",
                    "SRI_normalizer_curie": "MONDO:0005258",
                    "SRI_normalizer_name": "autism spectrum disorder",
                    "SRI_normalizer_category": "biolink:Disease",
                }
            }
        }
        mock_get.return_value = mock_response

        # When calling the synonymizer, yield the response
        result = normalize(curie="MONDO:01234", endpoint="http://dummy-endpoint")

        # Assert that None is returned when the curie is missing
        self.assertIsNone(result)
