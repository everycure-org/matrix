import pytest
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    StringType,
    StructType,
    StructField,
    IntegerType,
)

from matrix.pipelines.embeddings.nodes import compute_embeddings


@pytest.fixture(name="df")
def df_fixture(spark):
    return spark.createDataFrame(
        [(1, "Alice"), (2, "Bob"), (3, "Cathy"), (4, "Mark"), (5, "Pascal")],
        schema=StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("name", StringType(), False),
            ]
        ),
    )


def test_compute_embeddings(spark, df, mocker):
    def mock_batch(endpoint, model, api_key, batch):
        """Function to mock our batch method, returns array of arrays
        of the input elements."""
        return [[float(el)] for el in batch]

    # Ensures our batch method is mocked to return
    # a deterministic embedding on input for our unit test.
    func1_mock = mocker.patch(
        "matrix.pipelines.embeddings.nodes.batch", side_effect=mock_batch
    )

    # Given an input dataframe to compute embeddings for
    result = compute_embeddings(
        df, ["id"], "embedding", "dummy", 2, "dummy_endpoint", "model"
    ).orderBy("id")
    expected = spark.createDataFrame(
        [
            (1, [1.0], "Alice"),
            (2, [2.0], "Bob"),
            (3, [3.0], "Cathy"),
            (4, [4.0], "Mark"),
            (5, [5.0], "Pascal"),
        ],
        schema=StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("embedding", ArrayType(FloatType()), False),
                StructField("name", StringType(), False),
            ]
        ),
    )

    # Then result and expected dataframe equal
    assert sorted(result.collect()) == sorted(expected.collect())
