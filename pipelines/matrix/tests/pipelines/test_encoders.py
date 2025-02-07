from matrix.pipelines.embeddings.encoders import LangChainEncoder


def test_dummy():
    encoder = LangChainEncoder(dimensions=10, batch_size=2)
    result = list(encoder.apply(list(range(7))))
    print(result)
    assert False
