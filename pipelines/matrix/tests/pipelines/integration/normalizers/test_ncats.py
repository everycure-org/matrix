import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
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
        # Given input dataframe with single element
        (pd.DataFrame({"id": ["CHEBI:001"]}), ["CHEBI:normalized_001"]),
        # Given input dataframe with non existing element
        (pd.DataFrame({"id": ["CHEBI:foo"]}), [pd.NA]),
        # Given input dataframe with randomized order and non defined
        (
            pd.DataFrame({"id": ["CHEBI:002", "CHEBI:001", "CHEBI:foo"]}),
            ["CHEBI:normalized_002", "CHEBI:normalized_001", pd.NA],
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
    result_df = await normalizer.apply(input_df)

    # Then output dataframe is of correct structured, and correct
    # identifiers are normalized.
    assert "normalized_id" in result_df.columns
    assert result_df["normalized_id"].tolist() == expected_normalized_ids
