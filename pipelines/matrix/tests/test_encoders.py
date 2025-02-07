from unittest.mock import Mock

import pytest
from matrix.pipelines.embeddings.encoders import LangChainEncoder


@pytest.mark.parametrize(
    "input_data, mock_return",
    [
        (["mocked_embedding"] * 7, [1.0] * 7),
    ],
)
def test_dummy(input_data, mock_return):
    encoder = LangChainEncoder(dimensions=10, batch_size=2)
    encoder.apply = Mock(return_value=mock_return)
    result = list(encoder.apply(input_data))
    assert result == mock_return
