# Pandera Data Validation

Pandera is a powerful data validation library that we use extensively in our codebase for both pandas and PySpark DataFrames. This guide explains how to use Pandera validators effectively in our project.

## Import Conventions

In our codebase, we follow strict import conventions for clarity and maintainability:

```python
# Preferred imports
import pandera  # not 'as pa'
import pandera.pyspark  # not 'as py'

# Not recommended
import pandera as pa  # avoid abbreviations
import pandera.pyspark as py  # avoid abbreviations
```

We avoid using abbreviations like `pa` or `py` as they can cause confusion, especially when working with multiple DataFrame types and validation approaches.

## DataFrame Types and Validation

`pyspark.sql.DataFrame` and `pyspark.DataFrame` are effectively the same class.

Our codebase works with several types of DataFrames, each requiring different validation approaches:

# TODO: Differentiate use cases between two different types of PySpark DataFrames, and provide examples + justification from our codebase.


Subsequent questions:
> - The output of schema.validate will produce a dataframe in pyspark SQL even in case of errors during validation. Instead of raising the error, the errors are collected and can be accessed via the dataframe.pandera.errors attribute as shown in this example. This design decision is based on the expectation that most use cases for pyspark SQL dataframes means entails a production ETL setting. In these settings, pandera prioritizes completing the production load and saving the data quality issues for downstream rectification.
> - There is no support for lambda based vectorized checks since in spark lambda checks needs UDFs, which is inefficient. However pyspark sql does support custom checks via the register_check_method() decorator.
> - Unlike the pandera pandas schemas, the default behaviour of the pyspark SQL version for errors is lazy=True, i.e. all the errors would be collected instead of raising at first error instance.
> - In defining the type annotation, there is limited support for default python data types such as int, str, etc. When using the pandera.pyspark API, using pyspark.sql.types based datatypes such as StringType, IntegerType, etc. is highly recommended.



# TODO: Integrate this part of documentation

> Granular Control of Pandera’s Execution
> new in 0.16.0
> By default, error reports are generated for both schema and data level validation. Adding support for pysqark SQL also comes with more granular control over the execution of Pandera’s validation flow.
> 
> This is achieved by introducing configurable settings using environment variables that allow you to control execution at three different levels:
>
>SCHEMA_ONLY: perform schema validations only. It checks that data conforms to the schema definition, but does not perform any data-level validations on dataframe.
>
>DATA_ONLY: perform data-level validations only. It validates that data conforms to the defined checks, but does not validate the schema.
>
>SCHEMA_AND_DATA: (default) perform both schema and data level validations. It runs most exhaustive validation and could be compute intensive.
>
>You can override default behaviour by setting an environment variable from terminal before running the pandera process as:
>
>export PANDERA_VALIDATION_DEPTH=SCHEMA_ONLY

# TODO: Add section about validation disabling 

> Switching Validation On and Off
> new in 0.16.0
> 
> It’s very common in production to enable or disable certain services to save computing resources. We thought about it and thus introduced a switch to enable or disable pandera in production.
> 
> You can override default behaviour by setting an environment variable from terminal before running the pandera process as follow:
> 
> export PANDERA_VALIDATION_ENABLED=False
> This will be picked up by pandera to disable all validations in the application.
> 
> By default, validations are enabled and depth is set to SCHEMA_AND_DATA which can be changed to SCHEMA_ONLY or DATA_ONLY as required by the use case.




1. **Pandas DataFrames**: Local in-memory DataFrames, best for smaller datasets and complex validations. Supports the full range of Pandera validations including statistical checks and complex data type validations.

2. **PySpark DataFrames**: Distributed DataFrames that can handle large-scale data. Uses Spark's native type system and offers more limited validation capabilities but better performance. Importantly, t

TODO: Verify that PySpark DataFrame validations are executed on the driver node.

## Class-Based Schema Definition

We prefer using class-based API over the object-based API for better code organization and type hints. Here's how to define and use them:

# TODO: Replace this with an actual example from our codebase.

```python
import pandera
from pandera.typing import Series

class UserSchema(pandera.SchemaModel):
    id: Series[int] = pandera.Field(nullable=False, unique=True)
    name: Series[str] = pandera.Field(nullable=False)
    age: Series[int] = pandera.Field(ge=0, lt=150)
    email: Series[str] = pandera.Field(str_matches=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    class Config:
        strict = True
        coerce = True
```

## Pandas vs PySpark Validation

Our codebase supports both Pandas and PySpark DataFrame validation, but there are important differences:

### Pandas Validation

# TODO: Replace this with an actual example from our codebase.
```python
import pandera

# Define schema for pandas
schema = pandera.DataFrameSchema({
    "float_col": pandera.Column(float, pandera.Check.ge(10.0)),
    "int_col": pandera.Column(str),
    "string_col": pandera.Column(str, nullable=False)
})
```

### PySpark Validation

# TODO: Replace this with an actual example from our codebase.

```python
import pandera.pyspark
import pyspark.sql.types as T

# Define schema for PySpark
schema = pandera.pyspark.DataFrameSchema({
    "int_col": pandera.pyspark.Column(T.IntegerType()),
    "float_col": pandera.pyspark.Column(T.FloatType(), pandera.pyspark.Check.ge(10.0)),
    "string_col": pandera.pyspark.Column(T.StringType())
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
schema = pandera.DataFrameSchema(
    {
        "id1": pandera.Column(int, nullable=False),  # composite primary key
        "id2": pandera.Column(str, nullable=False),  # composite primary key
        "name": pandera.Column(str, nullable=True),
    },
    unique=["id1", "id2"]  # checks joint uniqueness
)
```

### Multiple Validations
```python
schema = pandera.DataFrameSchema({
    "string_col": pandera.Column(
        str,
        [
            pandera.Check.str_matches(r"^[A-Z]"),
            pandera.Check.isin(["Asia", "Africa", "Europe"]),
            pandera.Check(lambda x: len(x) < 20),
        ],
    )
})
```

### Regex Pattern Matching
```python
schema = pandera.DataFrameSchema({
    ".*_col": pandera.Column(nullable=False, regex=True)  # matches any column ending with '_col'
})
```

## Best Practices

1. Always use class-based schemas for new code
2. Include clear error messages in custom checks
3. Use composite keys when dealing with multi-column uniqueness
4. Consider performance implications when validating large PySpark DataFrames
5. Add schema validation tests for critical data pipelines
6. Use full import names (`pandera`, not `pa`) for clarity
7. Consider DataFrame type when choosing validation approach

For more examples, see:
# TODO: Add examples from our codebase.

# TODO: Add considerations for uniqueness constraints in Pandera's distributed (SQL) PySpark DataFrames.
