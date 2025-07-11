from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from matrix.pipelines.integration.normalizers.ncats import NCATSNodeNormalizer


class AsyncMock(MagicMock):
    async def json(self, *args, **kwargs):
        return {
            "CHEBI:001": {
                "id": {"identifier": "CHEBI:normalized_001"},
                "type": ["biolink:ChemicalEntity", "biolink:NamedThing"],
            },
            "CHEBI:002": {"id": {"identifier": "CHEBI:normalized_002"}, "type": ["biolink:ChemicalEntity"]},
        }


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
@pytest.mark.parametrize(
    "input_df,expected_normalized_results",
    [
        # Normal Case: Single known ID
        (
            ["CHEBI:001"],
            [
                {
                    "normalized_id": "CHEBI:normalized_001",
                    "normalized_categories": ["biolink:ChemicalEntity", "biolink:NamedThing"],
                }
            ],
        ),
        # Normal Case: single unknown ID
        (["CHEBI:foo"], [{"normalized_id": None, "normalized_categories": ["biolink:NamedThing"]}]),
        # Given input list with a mix of known and unknown IDs
        (
            ["CHEBI:002", "CHEBI:001", "CHEBI:foo"],
            [
                {
                    "normalized_id": "CHEBI:normalized_002",
                    "normalized_categories": [
                        "biolink:SmallMolecule",
                        "biolink:MolecularEntity",
                        "biolink:ChemicalEntity",
                        "biolink:NamedThing",
                    ],
                },
                {
                    "normalized_id": "CHEBI:normalized_001",
                    "normalized_categories": ["biolink:ChemicalEntity", "biolink:NamedThing"],
                },
                {"normalized_id": None, "normalized_categories": ["biolink:NamedThing"]},
            ],
        ),
    ],
)
async def test_apply(mock_post, input_df, expected_normalized_results):
    # Given an instance of the NCATSNodeNormalizer
    normalizer = NCATSNodeNormalizer("http://mock-endpoint.com", True, True)

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_post.return_value.__aenter__.return_value = mock_response

    # When applying the apply
    result = await normalizer.apply(input_df)

    # Then output is of correctly structured, and correct identifiers are normalized.
    assert result == expected_normalized_results


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
async def test_missing_categories(mock_post):
    # Returns no 'type' to simulate missing categories
    # Note that this should never happen in NN but a precaution
    mock_post.return_value.__aenter__.return_value.json = AsyncMock(
        return_value={"CHEBI:003": {"id": {"identifier": "CHEBI:normalized_003"}}}
    )
    mock_post.return_value.__aenter__.return_value.status = 200

    normalizer = NCATSNodeNormalizer("http://mock-endpoint.com", True, True)
    result = await normalizer.apply(["CHEBI:001"])
    # Should fallback to biolink:NamedThing
    assert result == [{"normalized_id": "CHEBI:normalized_003", "normalized_categories": ["biolink:NamedThing"]}]
