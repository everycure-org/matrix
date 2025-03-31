from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from matrix.pipelines.integration.normalizers.ncats import NCATSNodeNormalizer


class AsyncMock(MagicMock):
    async def json(self, *args, **kwargs):
        return {
            "CHEBI:001": {"id": {"identifier": "CHEBI:normalized_001"}},
            "CHEBI:002": {"id": {"identifier": "CHEBI:normalized_002"}},
        }


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
@pytest.mark.parametrize(
    "input_df,expected_normalized_ids",
    [
        # Given input list with single element
        (["CHEBI:001"], ["CHEBI:normalized_001"]),
        # Given input list with non-existing element
        (["CHEBI:foo"], [None]),
        # Given input list with randomized order and non defined
        (
            ["CHEBI:002", "CHEBI:001", "CHEBI:foo"],
            ["CHEBI:normalized_002", "CHEBI:normalized_001", None],
        ),
    ],
)
async def test_apply(mock_post, input_df, expected_normalized_ids):
    # Given an instance of the NCATSNodeNormalizer
    normalizer = NCATSNodeNormalizer("http://mock-endpoint.com", True, True)

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_post.return_value.__aenter__.return_value = mock_response

    # When applying the apply
    result = await normalizer.apply(input_df)

    # Then output is of correctly structured, and correct identifiers are normalized.
    assert result == expected_normalized_ids
