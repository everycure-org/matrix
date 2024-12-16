# Pandera Data Validation

Pandera is a powerful data validation library that we use extensively in our codebase for both pandas and PySpark DataFrames. This guide explains how to use Pandera validators effectively in our project.

## Class-Based Schema Definition

We prefer using class-based schema definitions for better code organization and type hints. Here's how to define and use them:

```python
import pandera as pa
from pandera.typing import Series

class UserSchema(pa.SchemaModel):
    id: Series[int] = pa.Field(nullable=False, unique=True)
    name: Series[str] = pa.Field(nullable=False)
    age: Series[int] = pa.Field(ge=0, lt=150)
    email: Series[str] = pa.Field(str_matches=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    class Config:
        strict = True
        coerce = True
```

## Pandas vs PySpark Validation

Our codebase supports both Pandas and PySpark DataFrame validation, but there are important differences:

### Pandas Validation
```python
import pandera as pa

# Define schema for pandas
schema = pa.DataFrameSchema({
    "float_col": pa.Column(float, pa.Check.ge(10.0)),
    "int_col": pa.Column(str),
    "string_col": pa.Column(str, nullable=False)
})
```

### PySpark Validation
```python
import pandera.pyspark as py
import pyspark.sql.types as T

# Define schema for PySpark
schema = py.DataFrameSchema({
    "int_col": py.Column(T.IntegerType()),
    "float_col": py.Column(T.FloatType(), py.Check.ge(10.0)),
    "string_col": py.Column(T.StringType())
})
```

Key differences:
- PySpark uses native Spark types (e.g., `T.IntegerType()`)
- Pandas uses Python types (e.g., `int`, `str`)
- PySpark validation is generally less flexible but more performant
- Pandas validation offers more complex checks and statistical validations

## Examples from Our Codebase

### Composite Primary Keys
```python
schema = pa.DataFrameSchema(
    {
        "id1": pa.Column(int, nullable=False),  # composite primary key
        "id2": pa.Column(str, nullable=False),  # composite primary key
        "name": pa.Column(str, nullable=True),
    },
    unique=["id1", "id2"]  # checks joint uniqueness
)
```

### Multiple Validations
```python
schema = pa.DataFrameSchema({
    "string_col": pa.Column(
        str,
        [
            pa.Check.str_matches(r"^[A-Z]"),
            pa.Check.isin(["Asia", "Africa", "Europe"]),
            pa.Check(lambda x: len(x) < 20),
        ],
    )
})
```

### Regex Pattern Matching
```python
schema = pa.DataFrameSchema({
    ".*_col": pa.Column(nullable=False, regex=True)  # matches any column ending with '_col'
})
```

## Best Practices

1. Always use class-based schemas for new code
2. Include clear error messages in custom checks
3. Use composite keys when dealing with multi-column uniqueness
4. Consider performance implications when validating large PySpark DataFrames
5. Add schema validation tests for critical data pipelines

For more examples, see:
- `pipelines/matrix/packages/refit/src/refit/tests/v1/test_validator.py`
- `pipelines/matrix/tests/pipelines/test_preprocessing.py` 