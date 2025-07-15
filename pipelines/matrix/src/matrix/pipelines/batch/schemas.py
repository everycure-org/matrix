import pyarrow as pa
from pyspark.sql.pandas.types import from_arrow_type
from pyspark.sql.types import StructType

embeddings: pa.lib.Schema = pa.schema({"key": pa.string(), "value": pa.list_(pa.float32()), "api": pa.string()})

node_normalizer: pa.lib.Schema = pa.schema(
    {
        "key": pa.string(),
        "value": pa.struct({"normalized_id": pa.string(), "normalized_categories": pa.list_(pa.string())}),
        "api": pa.string(),
    }
)


def to_spark_schema(s: pa.lib.Schema) -> StructType:
    # to be removed when project used Spark4.x, which supports conversion
    # between pyarrow tables and spark dataframes out of the box.
    # See https://github.com/apache/spark/pull/46529/files
    result = StructType()
    for field in s:
        result.add(field.name, from_arrow_type(field.type))
    return result
