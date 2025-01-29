from unittest.mock import Mock

from matrix.pipelines.embeddings.encoders import LangChainEncoder


def test_dummy():
    mock = Mock()
    encoder = LangChainEncoder(dimensions=10, encoder=mock, batch_size=2)
    result = list(encoder.apply(list(range(7))))
    print(result)
    assert False
