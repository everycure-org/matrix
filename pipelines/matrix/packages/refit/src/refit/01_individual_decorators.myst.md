---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
```{code-cell}
:tags: ["hide-input"]
import sys
sys.path.insert(0,'../')
sys.path.insert(0,'packages')
```

```{code-cell}
:tags: ["hide-input"]
import os

if os.environ.get("CIRCLECI"):
    python_executable = os.environ.get("VENV_PYTHON_EXEC")
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable
    os.environ["PYSPARK_PYTHON"] = python_executable

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
```
# Example Usage

Described below are available decorators and code examples on how to use them.

## Inject Object

Keyword: `object`

The inject object decorator will handle any incoming dictionaries and parse them for
`object` definitions, before passing it to the function. When dealing with pipelines
such as `kedro`, `parameters.yml` is where parameters may be defined. In YAML for example,
objects cannot be defined, but this way, one can define objects declaratively in YAML,
keeping the underlying code cleaner.

The key principle here is dependency injection, or inversion of control. According to
this principle, a class (or function) should concentrate on fulfilling its
responsibilities and not on creating objects that it requires to fulfill those
responsibilities (https://www.freecodecamp.org/news/a-quick-intro-to-dependency-injection-what-it-is-and-when-to-use-it-7578c84fa88f/).

Let's demonstrate the impact this has on the code and how you can call a function:
```
def bad_fit_data(
    data: pd.DataFrame, X: List[str], y: str, model_str: str
) -> BaseEstimator:
    """Fit a model on data.

    Args:
        data: A pandas dataframe.
        X: A list of features.
        y: The column name of the target.
        model_str: The string path to the model object.

    Returns:
        A fitted model object.
    """
    model_object = load_obj(model_str)
    fitted_model_object = model_object.fit(data[X], data[y])
    return fitted_model_object

# calling the "bad" example:
bad_fitted_model_object = bad_fit_data(
    data=df,
    X=["feature_1"],
    y=["target"],
    model="sklearn.linear_model.LinearRegression",
)
```
```{code-cell}
# let's define a cleaner example
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator

from refit.v1.core.inject import inject_object


@inject_object()
def fit_data(
    data: pd.DataFrame, X: List[str], y: str, model_object: BaseEstimator
) -> BaseEstimator:
    """Fit a model on data.

    Args:
        data: A pandas dataframe.
        X: A list of features.
        y: The column name of the target.
        model_object: The sklearn model type to use.

    Returns:
        A fitted model object.
    """
    fitted_model_object = model_object.fit(data[X], data[y])
    return fitted_model_object

df = pd.DataFrame(
        [
            {"feature_1": 1, "target": 1},
            {"feature_1": 2, "target": 2},
            {"feature_1": 3, "target": 3},
        ]
    )

from sklearn.linear_model import LinearRegression

# usage as normal in code
fitted_model_object = fit_data(
    data=df, X=["feature_1"], y=["target"], model_object=LinearRegression()
)

# parametrised via dictionary to be used in pipeline
fitted_model_object = fit_data(
    data=df,
    X=["feature_1"],
    y=["target"],
    model_object={"object": "sklearn.linear_model.LinearRegression"},
)
```



### Example
The example function:
```{code-cell}
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute._base import _BaseImputer

from refit.v1.core.inject import inject_object


@inject_object()
def fit_then_predict(data: pd.DataFrame, imputer: _BaseImputer) -> pd.DataFrame:
    imputed_data = imputer.fit_transform(data)

    return pd.DataFrame(imputed_data, columns=data.columns)
```

The dataset:
```{code-cell}
data = pd.DataFrame([{"c1": 1}, {"c1": None}])

print(data)
```

#### Example 1 - Running Simple Imputer
Running using code:
```{code-cell}
imputed_data = fit_then_predict(data=data, imputer=SimpleImputer())
print(imputed_data)
```
The decorator has no effect.


Running using parameters:
```{code-cell}
imputed_data = fit_then_predict(
    data=data, imputer={"object": "sklearn.impute.SimpleImputer"}
)
print(imputed_data)
```


#### Example 2 - Running KNNImputer
Running using code:
```{code-cell}
imputed_data = fit_then_predict(data=data, imputer=KNNImputer(n_neighbors=2))
print(imputed_data)
```

Running using parameters:
```{code-cell}
imputed_data = fit_then_predict(
    data=data, imputer={"object": "sklearn.impute.KNNImputer", "n_neighbors": 2}
)
print(imputed_data)
```
Notice that kwargs can be declared within the dictionary.



### Further Usage Examples

The decorator will scan any dictionary passed to a function for the
keywords: `{"object": "path.to.definition"}`.

For classes, init arguments may be passed like so:

    {
        "object": "path.to.MyClass",
        "class_arg1": "value",
        "class_arg2": "another_value"
    }

This is equivalent to
`MyClass(class_arg1="value", "class_arg2="another_value")` assuming:

    class MyClass:
        def __init__(class_arg1, class_arg2):
           ...

The keyword `instantiate` may be used to prevent the decorator from
instantiating the class:

    {
        "object": "path.to.MyClass",
        "instantiate": False
    }

This is equivalent to `MyClass` (or `x = MyClass`) without the
parantheses which you can instantiate later in your own code. If
`instantiate: False` is supplied, class init arguments have no effect.
If the `instantiate: False` argument is supplied, it is recommended to
remove init arguments.

To parametrize for functions, simply provide the path to the function:

    {
        "object": "path.to.my_function"
    }

This is equivalent to `x = my_function`.

Passing an argument to a function defined as such will cause the
function to be evaluated:

    def my_function(x):
        return x

    {
        "object": "path.to.my_function",
        "x": 1
    }

This is equivalent to `result = my_function(x=1)`. Additional arguments
should be passed via use of keywords, corresponding to actual function
signature.

Another example:

    {
        "object": "pyspark.sql.functions.max",
        "col": "column_1"
    }

This is equivalent to `f.max("column_1")`.

### Advanced Usage Examples

The decorator also handles nested structures (via recursion):

    # nested class example where objects defined need other objects
    {
        "object": "path.to.MyClass":
        "some_init_arg_that_requires_another_object": {
            "object": "path.to.that.Object",
            "another_init_that_requires_yet_another_object": {
                "object": "path.to.yet.another.Object"
            }
        }
    }

    # nested function example where function requires an object.\
    {
        "object": "path.to.my_func":
        "my_func_args": {
            "object": "path.to.a.Class"
        }
    }

The recommended rule of thumb is to look at the original code, make sure
that is clean and easy to read. In most cases, if the input parameter is
heavily nested, this might indicate the original code could benefit from
refactoring to flatten out logic.

The decorator allows us to exclude certain keywords if we want to delay their injection.
To do this you would use the `exclude_kwargs` parameter. This expects a list of keywords to exclude.

To see this in action, consider a situation where you are calling a function that needs to serialize some
parameters that also use `refit` syntax.

    @inject_object(exclude=['params_to_serialize'])
    def serialize_object(writer: AbstractWriter, params_to_serialize: Dict[str, str]):
      writer.write(path, params_to_serialize)

We can call this function using the following config:

    {
        "writer": {
            "object": "path.to.some.abstract.writer"
            "save_path": "some_path"
            "compress": True
        },
        "params_to_serialize": {
            "object": "path.to.MyClass":
            "some_init_arg_that_requires_another_object": {
                "object": "path.to.that.Object",
                "another_init_that_requires_yet_another_object": {
                    "object": "path.to.yet.another.Object"
                }
            }
        }
    }

In this instance, only the `AbstractWriter` class will be injected, whilst inside the function,

    params_to_serialize :Dict = {
        "object": "path.to.MyClass":
        "some_init_arg_that_requires_another_object": {
            "object": "path.to.that.Object",
            "another_init_that_requires_yet_another_object": {
                "object": "path.to.yet.another.Object"
            }
        }
    }

## Inline `has_schema`

Inline style `has_schema`:
```{code-cell}
:tags: ["hide-cell"]
from refit.v1.core.inline_has_schema import has_schema


import datetime
import time

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)



schema = StructType(
    [
        StructField("int_col", IntegerType(), True),
        StructField("long_col", LongType(), True),
        StructField("string_col", StringType(), True),
        StructField("float_col", FloatType(), True),
        StructField("double_col", DoubleType(), True),
        StructField("date_col", DateType(), True),
        StructField("datetime_col", TimestampType(), True),
        StructField("array_int", ArrayType(IntegerType()), True),
    ]
)

data = [
    (
        1,
        2,
        "awesome string",
        10.01,
        0.89,
        pd.Timestamp("2012-05-01").date(),
        datetime.datetime(2023,1,1),
        [1, 2, 3],
    ),
     (
        2,
        2,
        None,
        10.01,
        0.89,
        pd.Timestamp("2012-05-01").date(),
        datetime.datetime(2023,1,1),
        [1, 2, 3],
    ),
]

spark = (
    SparkSession.builder.config("spark.ui.showConsoleProgress", False)
    .config("spark.sql.shuffle.partitions", 1)
    .getOrCreate()
)

spark_df = spark.createDataFrame(data, schema)
```

Assuming we have the following dataframe:
```{code-cell}
spark_df.show(truncate = False)
```

And assuming we have the following function:
```{code-cell}
import pyspark.sql.functions as f

@has_schema(
    schema={
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
        "new_output": "int",
    },
    allow_subset=True,
    raise_exc=True,
    relax=False,
)
def my_func(df):
    df_new = df.withColumn("new_output", f.lit(1))
    return df_new
```

We can run our function like so and the schema checking will be performed on the
output by default:
```{code-cell}
result = my_func(spark_df)
result.show(truncate = False)
```

The decorator works with both pandas and spark dataframes.
```{code-cell}
:tags: ["hide-input"]
pd_df = pd.DataFrame(
    {
        "float_col": [1.0, 2.0],
        "int_col": [1, 2],
        "datetime_col": [pd.Timestamp("20180310"), pd.Timestamp("20180410")],
        "date_col": [pd.Timestamp("20180310").date(), pd.Timestamp("20180410").date()],
        "string_col": ["foo", "bar"],
    }
)

pd_df["datetime_ms_col"] = pd_df["datetime_col"].values.astype("datetime64[ms]")
```

The pandas dataframe:
```{code-cell}
print(pd_df)
```

Assuming we have the following function:

```{code-cell}
@has_schema(
    schema={
        "int_col": "int64",
        "string_col": "object",
        "float_col": "float64",
        "date_col": "object",
        "datetime_col": "datetime64[ns]",
        "new_output": "int64",
        "datetime_ms_col": "datetime64[ns]",
    },
    allow_subset=False,
    raise_exc=True,
)
def pandas_example(df):
    df["new_output"] = 1
    return df
```

Running gives us:
```{code-cell}
result = pandas_example(pd_df)
print(result)
```

Below is another example for multiple inputs:
```{code-cell}


@has_schema(
    df="df1",
    schema={
        "int_col": "int64",
    },
    allow_subset=True,
    raise_exc=False,
)
@has_schema(
    df="df2",
    schema={
        "int_col": "int64",
    },
    allow_subset=True,
    raise_exc=False,
)
def multi_input_example_func(df1, df2):
    return True

result = multi_input_example_func(pd_df, pd_df)
print(result)
```


## Has Schema

Check the input schema of a dataframe, but with keyword injection of instead of the
Inline style.

```{code-cell}

import pandera as pa
import pandera.pyspark as py
import pyspark
import pyspark.sql.types as T
import pytest

from refit.v1.core.has_schema import has_schema


@has_schema()
def node_func(df):
    return df

_input_has_schema = {
    "df": "df",
    "expected_schema": {
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
    },
    "allow_subset": False,
    "raise_exc": True,
    "relax": False,
}
_output_has_schema = {
    "expected_schema": {
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
    },
    "allow_subset": False,
    "raise_exc": True,
    "relax": False,
    "output":0,
}

node_func(
    df=spark_df,
    input_has_schema=_input_has_schema,
    output_has_schema=_output_has_schema
)

```

For multiple inputs or outputs, pass in a list of dictionaries instead:
```{code-cell}
@has_schema()
def node_func(df1, df2):
    return df1, df2

_input_has_schema = [
    {
        "df": "df1",
        "expected_schema": {
            "int_col": "int",
            "long_col": "bigint",
            "string_col": "string",
            "float_col": "float",
            "double_col": "double",
            "date_col": "date",
            "datetime_col": "timestamp",
            "array_int": "array<int>",
        },
        "allow_subset": False,
        "raise_exc": True,
        "relax": False,
    },
    {
        "df": "df2",
        "expected_schema": {
            "int_col": "int",
            "long_col": "bigint",
            "string_col": "string",
            "float_col": "float",
            "double_col": "double",
            "date_col": "date",
            "datetime_col": "timestamp",
            "array_int": "array<int>",
        },
        "allow_subset": False,
        "raise_exc": True,
        "relax": False,
    }
]
_output_has_schema = [
    {
        "expected_schema": {
            "int_col": "int",
            "long_col": "bigint",
            "string_col": "string",
            "float_col": "float",
            "double_col": "double",
            "date_col": "date",
            "datetime_col": "timestamp",
            "array_int": "array<int>",
        },
        "allow_subset": False,
        "raise_exc": True,
        "relax": False,
        "output": 0,
    },
    {
        "expected_schema": {
            "int_col": "int",
            "long_col": "bigint",
            "string_col": "string",
            "float_col": "float",
            "double_col": "double",
            "date_col": "date",
            "datetime_col": "timestamp",
            "array_int": "array<int>",
        },
        "allow_subset": False,
        "raise_exc": True,
        "relax": False,
        "output": 1,
    }
]
node_func(
    df1=spark_df,
    df2=spark_df,
    input_has_schema=_input_has_schema,
    output_has_schema=_output_has_schema
)
```



If your function returns a dictionary dataframe, you can also reference output
by string:
```{code-cell}
@has_schema()
def node_func(df1, df2):
    return {"key1": df1, "key2": df2}

_output_has_schema = [
    {
        "expected_schema": {
            "int_col": "int",
            "long_col": "bigint",
            "string_col": "string",
            "float_col": "float",
            "double_col": "double",
            "date_col": "date",
            "datetime_col": "timestamp",
            "array_int": "array<int>",
        },
        "allow_subset": False,
        "raise_exc": True,
        "relax": False,
        "output": "key1",
    },
    {
        "expected_schema": {
            "int_col": "int",
            "long_col": "bigint",
            "string_col": "string",
            "float_col": "float",
            "double_col": "double",
            "date_col": "date",
            "datetime_col": "timestamp",
            "array_int": "array<int>",
        },
        "allow_subset": False,
        "raise_exc": True,
        "relax": False,
        "output": "key2",
    }
]
node_func(
    df1=spark_df,
    df2=spark_df,
    output_has_schema=_output_has_schema
)
```

## Retry

The retry decorator retries a function given a list of exceptions. This may be useful
fo functions that are calling APIs.

```{code-cell}
from refit.v1.core.retry import retry

@retry()
def add_retry_column(*args, **kwargs):
    def add_column(df):
        df["retry_col"] = 1
        return df
    return add_column(*args, **kwargs)

@retry()
def dummy_func_raises(*args, **kwargs):
    def dummy_raises(x):
        print("x is", x)
        raise TypeError("random TypeError")
    return dummy_raises(*args, **kwargs)
```
Running the following should show the behavior of the decorator, when the expected error
is raised you see the statement printed for each retry.

```{code-cell}
dummy_func_raises(
    retry={"exception": [TypeError], "max_tries": 5, "interval": 0.01},
    x=2
)
```
Since no error is raised in the following snippet, the decorator doesn't retry.
```{code-cell}
result = add_retry_column(
    retry={"exception": [TypeError], "max_tries": 5, "interval": 0.01},
    df=pd_df
)
print(result)
```

## Input kwarg filter

This decorator modifies the function definition to include an additional filter based on input kwargs.
This works for both spark and pandas dataframes.

A node that does purely a filter and does IO is a low value node. There is cost from
maintaining extra entries in the catalog and pipeline. However, converting the dataset
(if kedro) to a `MemoryDataSet` breaks integration, as nodes should be re-runnable on
their own.

An alternative is to modify the source code to add a filter, but this also dilutes
the original logic, making it harder to read.

```{code-cell}
from refit.v1.core.input_kwarg_filter import add_input_kwarg_filter

@add_input_kwarg_filter()
def my_node_func(df):
    return df

spark_df.count()

result1 = my_node_func(df=spark_df, kwarg_filter={"df":"int_col != 1"})

result1.count()

print(pd_df.head())

result2 = my_node_func(df=pd_df, kwarg_filter={"df":"string_col != 'foo'"})

print(result2.head())
```

## Input kwarg select

This decorator modifies the function definition to include an additional select based on input kwargs.
This works only for dataframes.

A node that does purely a select and does IO is a low value node. There is cost from
maintaining extra entries in the catalog and pipeline. However, converting the dataset
(if kedro) to a `MemoryDataSet` breaks integration, as nodes should be re-runnable on
their own.

An alternative is to modify the source code to add a select, but this also dilutes
the original logic, making it harder to read.

```{code-cell}
from refit.v1.core.input_kwarg_select import add_input_kwarg_select

@add_input_kwarg_select()
def my_node_func(df):
    return df

spark_df.show()

result1 = my_node_func(df=spark_df, kwarg_select={"df":["length(string_col)"]})

result1.show()
```


## Unpack params
This decorator unpacks the top level dictionaries in args by 1 level.
Most beneficial if used as part of a kedro node.


```{code-cell}
from refit.v1.core.unpack import unpack_params

@unpack_params()
def my_func_pd(*args, **kwargs):
    def dummy_func(df, x, y):
        df["unpack"] = df["int_col"] + x + y
        return df

    return dummy_func(*args, **kwargs)


result = my_func_pd(unpack={"df": pd_df, "x": 1, "y": 0})

print(result)
```
The decorator unpacks the params and we get the expected result.


One can remove the unpack keyword when not using as part of the
node decorator.

```{code-cell}
try:
   result = my_func_pd({"df": pd_df, "x": 1, "y": 0})

except TypeError as e:
    print(e)

```

When using the above with kedro nodes, it would be as follows:
Unpacks dictionary with key is `unpack`. Typically used to unpack a dictionary from
kedro parameters.

Example usage:
```{code-cell}
# Unpack using args
@unpack_params()
def dummy_func(df, x, y, z):
    df["test"] = x + y - z
    return df

x = 1
param = {"unpack": {"y": 1, "z": 2}}
result = dummy_func(df, x, param)

# Unpack using kwargs
params = {"x": 1, "y": 1, "z": 2}
result = dummy_func(df=df, unpack=params)

# Unpack using args and kwargs
@unpack_params()
def dummy_func3(df, param1, param2, x, y, z,):
    df["col2"] = param1["col2"]
    df["col3"] =  param2["col3"]
    df["col4"] = x+y-z
    return df
params1 = {"param1":{"col2": 1}, "unpack": {"y": 2}}
params2 = {"param2":{"col3": 2}, "unpack": {"z": 3}}
result = dummy_func3(df, params1, params2, unpack={"x": 1})
```

## Output filter

This decorator modifies the function definition to include an additional filter to be applied on output dataframe. This works for both spark and pandas dataframes.
```{code-cell}
from refit.v1.core.output_filter import add_output_filter

@add_output_filter()
def my_node_func(df):
    return df

spark_df.count()

result1 = my_node_func(df=spark_df, output_filter="int_col != 1")

result1.count()

print(pd_df.head())

result2 = my_node_func(df=pd_df, output_filter="string_col != 'foo'")

print(result2.head())
```

## Output primary key check

This decorator performs the primary key functionality (duplicate and not null checks) on
the set of columns we pass from an output dataframe (spark and pandas).

Below example shows the primary key check without allowing any null and
duplicate values in it. This can be achieved by setting `nullable = False`.
By default `nullable` option is False.

```{code-cell}
from refit.v1.core.output_primary_key import add_output_primary_key

@add_output_primary_key()
def my_node_func(df):
    return df

spark_df.show()

try:
   result1 = my_node_func(df=spark_df, output_primary_key={"columns": ["int_col", "string_col"]})

except TypeError as error:
    print(error)

print(pd_df.head())

result2 = my_node_func(df=pd_df, output_primary_key={"columns": ["int_col"]})

print(result2.head())
```

Now check by setting option `nullable=True`, this allows null in composite key and without duplicate values in it.
However for single column of primary key, it won't allow duplicates and null values in it.

```{code-cell}

@add_output_primary_key()
def my_node_func(df):
    return df

spark_df.show()

result1 = my_node_func(df=spark_df, output_primary_key={"nullable":True, "columns": ["int_col", "string_col"]})

result1.show()

print(pd_df.head())

result2 = my_node_func(df=pd_df, output_primary_key={"nullable":True, "columns": ["int_col", "string_col"]})

print(result2.head())
```


## Refit Style Primary Key Check

This decorator is the `refit` style primary key check where a primary key
is defined as no duplicates and not nullable (by default). The decorator works for both
spark and pandas dataframes.

Note:
2. While using args use integer indices in input_primary_key specs.
3. While using kwargs use string indices in input_primary_key specs.
4. While returning list or tuple use integer indices in output_primary_key specs.
5. While returning dictionary use string indices in output_primary_key specs.
6. You can always switch and play between args-list, kwargs-list, args-dict, kwargs-dict type arg-return type combos.

Structure used for input_primary_key and output_primary_key is same:
```python
[
  {
    "columns": ["your primary key cols in this list"],
    "index": "key or index value",
    "nullable": True
  }
]
```

Following is an example showing kwargs used with dictionary return type:

```{code-cell}
from refit.v1.core.primary_key import primary_key

@primary_key()
def node_func(df1, df2, dict1):
    return {"df1": df1, "df2": df2, "dict1": dict1}

result_dict = node_func(df1=spark_df, df2=spark_df, dict1={"dummy_key": "dummy_value"},
input_primary_key=[{"columns": ["int_col"], "index": "df1"}, {"columns": ["int_col"], "index": "df2"}],
output_primary_key=[{"columns": ["int_col"], "index": "df1"}, {"columns": ["int_col"], "index": "df2"}])

spark_df.show()
result_dict["df1"].show()
result_dict["df2"].show()
```

Following is an example showing args used with list return type:
```{code-cell}
from refit.v1.core.primary_key import primary_key

@primary_key()
def node_func(df1, df2, dict1):
    return df1, df2, dict1

df1, df2, dict1 = node_func(spark_df, spark_df, {"dummy_key": "dummy_value"},
input_primary_key=[{"columns": ["int_col"], "index": 0}, {"columns": ["int_col"], "index": 1}],
output_primary_key=[{"columns": ["int_col"], "index": 0}, {"columns": ["int_col"], "index": 1}])

spark_df.show()
df1.show()
df2.show()
```


## Inline primary key check

This decorator is the `inline` style primary key check  where a primary key
is defined as no duplicates and not nullable (by default). The decorator works for both
spark and pandas dataframes.

By default, the decorator will check the output dataframe's primary key:
```{code-cell}
from refit.v1.core.inline_primary_key import primary_key

@primary_key(
    primary_key=["int_col"],
)
def node_func(df):
    return df

result_df = node_func(df=spark_df)

spark_df.show()
result_df.show()
```

However, to check input dataframes, leverage the `df` argument:
```{code-cell}
@primary_key(
    df="my_df",
    primary_key=["int_col"],
)
def node_func(my_df):
    return my_df

result_df = node_func(my_df=spark_df)

spark_df.show()
result_df.show()
```

### Nullable Primary Key

Sometimes, when it comes to composite primary keys, we may want to allow nullables in
some of the columns (with the exception of all keys being null). We can do so using the
`nullable` argument:
```{code-cell}

@primary_key(
    primary_key=["int_col", "string_col"],
    nullable=True
)
def node_func(df):
    return df

result_df = node_func(df=spark_df)

spark_df.show()
result_df.show()

```

Note that by default `nullable` is False as we don't expect nulls in any part of a
primary key (composite or not). This function is particularly useful when it comes to
data exploration during the early stages.


### Function With Multiple Returned Variables
If a function returns multiple variables, you may leverage the `output` kwarg like so:
```{code-cell}
@primary_key(
    primary_key=["int_col", "string_col"],
    output=1, # or 0
    nullable=True
)
def node_func(df):
    return [df, df]

result_df1, result_df2 = node_func(df=spark_df)

spark_df.show()
result_df1.show()
result_df2.show()
```


## Make list regexable

Keyword: `No Keyword`

The usage is to decorate your core function (as opposed to being used with `@augment`).
This decorator allows you to provide a list of regex in your function when selecting columns from a dataframe
(pandas and spark).
We have implemented it defender style because this is more modifying the core function.
It's not something we want to be able to configure on the fly at node level.
It's conforming to the dependency inversion principle where you expect your required dependencies
(in this case a list of columns after regex from the dataframe).

The `@make_list_regexable` decorator gives you the ability to pass regex in your list when it comes to selecting columns from a dataframe.
The decorator requires two parameters, a source dataframe which has actual columns and a list which might contains
regex.
If the list is empty or no list is provided the decorator will not do any modifications, normal course will follow.
```{code-cell}
from refit.v1.core.make_list_regexable import make_list_regexable


@make_list_regexable(source_df = "df", make_regexable="param_keep_cols")
def accept_regexable_list(df, param_keep_cols, enable_regex):
    df = df[[*param_keep_cols]]
    return df

data = pd.DataFrame(
        [
            {"feature_1": 1, "target": 1},
            {"feature_1": 2, "target": 2},
            {"feature_1": 3, "target": 3},
        ]
    )

final_result = accept_regexable_list(
    df=data,
    param_keep_cols=["feature_.*"],
    enable_regex=True,
)

print(final_result)

```
## Fill nulls

This decorator modifies the function definition to fill the nulls with a
given value on the output dataframe. This works for both spark and
pandas dataframes.

Here is an example with a spark dataframe:
```{code-cell}
from refit.v1.core.fill_nulls import fill_nulls
@fill_nulls()
def my_node_func(df):
    return df
result1 = my_node_func(df=spark_df, fill_nulls={"value": "default_value", "column_list": ["string_col"]})
result1.show()
```

Here is an example with a pandas dataframe:
```{code-cell}
pd_df2 = pd.DataFrame(
    {
        "float_col": [1.0, 2.0, None],
        "int_col": [1, None, 2],
        "datetime_col": [pd.Timestamp("20180310"), pd.Timestamp("20180410"), pd.Timestamp("20180510")],
        "date_col": [pd.Timestamp("20180310").date(), pd.Timestamp("20180410").date(), pd.Timestamp("20180510").date()],
        "string_col": ["foo", "bar", None],
    }
)
print(pd_df2.head())
```
The column list can be a regex string, when `enable_regex` is set to True. eg: The following example takes all the columns.
In this case, pandas will fill in other columns because pandas allows mix types. This is likely
to cause other issues down the line, for example writing to parquet. Be careful when using the
regex functionality!

```{code-cell}
result2 = my_node_func(df=pd_df2, fill_nulls={"value": "default_value", "enable_regex": True, "column_list": [".*"]})
print(result2.head())
```

Here is another pandas example, which only fills the specified columns:
```{code-cell}
pd_df2 = pd.DataFrame(
    {
        "float_col": [1.0, 2.0, None],
        "int_col": [1, None, 2],
        "datetime_col": [pd.Timestamp("20180310"), pd.Timestamp("20180410"), pd.Timestamp("20180510")],
        "date_col": [pd.Timestamp("20180310").date(), pd.Timestamp("20180410").date(), pd.Timestamp("20180510").date()],
        "string_col": ["foo", "bar", None],
    }
)
result3 = my_node_func(df=pd_df2, fill_nulls={"value": 200, "column_list": ["int_col"]})
print(result3.head())
```

## Remove Debug Columns
This may be leveraged with parquet's columnar format, where the previous function
stores columns with `_` prefix to disk, but the next node only selects and passes
through columns without this `_` prefix. Using this pattern, we can choose to persist
more columns and be able to investigate if anything looks wrong without having to re-run
the node, yet not suffer from performance implications downstream.
Note that this efficiently depends on how the underlying dataframe library reads in parquet files.

**Note**: `remove_input_debug_columns=True` and `remove_output_debug_columns=True` needs to explicitly passed to the
decorator in order for this decorator to work.

### Refit approach
```{code-cell}
from refit.v1.core.remove_debug_columns import (
    remove_input_debug_columns,
    remove_output_debug_columns,
)
PREFIX = "_"

df = pd.DataFrame(
    columns=['col1','_col2','col3'],
    data = [
        (1,2,3)
    ]
)

@remove_input_debug_columns()
@remove_output_debug_columns()
def some_func1(df):
    df['col4']=4
    df['_col5']=5
    return df
res = some_func1(df, remove_input_debug_columns=True, remove_output_debug_columns=True)
print(res.head())
```

### Inline approach
```{code-cell}
from refit.v1.core.inline_remove_debug_columns import (
    remove_input_debug_columns,
    remove_output_debug_columns,
)
PREFIX = "_"

df = pd.DataFrame(
    columns=['col1','_col2','col3'],
    data = [
        (1,2,3)
    ]
)

@remove_input_debug_columns(remove_input_debug_columns=True)
@remove_output_debug_columns(remove_output_debug_columns=True)
def some_func1(df):
    df['col4']=4
    df['_col5']=5
    return df
res = some_func1(df)
print(res.head())
```


## Columns No Nulls
Originally created to check feature columns for non-nulls. If checking primary key,
simply use the primary key decorator with nullable set to false. 
```{code-cell}

from refit.v1.core.columns_no_nulls import columns_no_nulls

@columns_no_nulls()
def func1(x):
    return x

# single column
result = func1(x=spark_df, columns_no_nulls={"list_of_columns": ["int_col"]})

# multiple columns
result = func1(
    x=spark_df, columns_no_nulls={"list_of_columns": ["int_col", "float_col"]}
)

# if returning a list of dataframes
@columns_no_nulls()
def func2(x):
    return [x, x]

result = func2(
    x=spark_df,
    columns_no_nulls=[
        {"output": 0, "list_of_columns": ["int_col"]},
        {"output": 1, "list_of_columns": ["float_col"]},
    ],
)

# if returning a dictionary of dataframes
@columns_no_nulls()
def func3(x):
    return {"x": x, "y": x}

result = func3(
    x=spark_df,
    columns_no_nulls=[
        {"output": "x", "list_of_columns": ["int_col"]},
        {"output": "y", "list_of_columns": ["float_col"]},
    ],
)
```


## Validate

Validates the schema/data of input and output according to given schema with keyword injection.
Under the hood we are using Pandera's DataFrameSchema class that enables the specification of a 
schema that verifies the columns and index of a pandas/pyspark DataFrame object.
To read more about Pandera and DataFrameSchema class, 
visit: https://pandera.readthedocs.io/en/stable/index.html

### Pyspark dataframe validation
Validating the schema and data of spark dataframes 

```{code-cell}
:tags: ["hide-cell"]

schema = StructType(
    [
        StructField("int_col", IntegerType(), True),
        StructField("float_col", FloatType(), True),
        StructField("string_col", StringType(), True),
    ]
)

data = [
    (1, 2.0, "awesome string",),
    (2, 2.0, None,),
    (3, 2.0, "hello world",),
]

spark_df = spark.createDataFrame(data, schema)
```

```{code-cell}
spark_df.show()
```

```{code-cell}
from refit.v1.core.pandera import validate

@validate()
def node1_spark(input_df: pyspark.sql.DataFrame):
    return input_df

input_validation = {
    "input_df": py.DataFrameSchema(
        {
            "int_col": py.Column(T.IntegerType()),
            "float_col": py.Column(T.FloatType()),
            "string_col": py.Column(T.IntegerType()),
        },
    ),
}
output_validation = {
    "out": py.DataFrameSchema(
        {
            "int_col": py.Column(T.IntegerType()),
            "float_col": py.Column(T.FloatType(), py.Check.ge(10.0)),
            "string_col": py.Column(T.StringType()),
        },
    ),
}

raise_exec_on_input = False
raise_exec_on_output = False

df = node1_spark(
    input_df=spark_df,
    input_validation=input_validation,
    output_validation=output_validation,
    raise_exec_on_input=raise_exec_on_input,
    raise_exec_on_output=raise_exec_on_output,
)
```

### Pandas dataframe validation
Validating the schema and data for pandas dataframes 

```{code-cell}
:tags: ["hide-cell"]
data = [
    {"float_col": 1.0, "int_col": 1, "string_col": "foo",},
    {"float_col": 1.0, "int_col": 2, "string_col": "blabla",},
    {"float_col": 1.0, "int_col": 3, "string_col": None,},
]
pandas_df = pd.DataFrame(data)
```
```{code-cell}
print(pandas_df)
```
```{code-cell}
@validate()
def node1_pandas(input_df: pd.DataFrame):
    return input_df


input_validation = {
    "input_df": pa.DataFrameSchema(
        {
            "float_col": pa.Column(float, pa.Check.ge(10.0)),
            "int_col": pa.Column(str),
            "string_col": pa.Column(str),
        }
    )
}

output_validation = {
    "out": pa.DataFrameSchema(
        {
            "float_col": pa.Column(float, pa.Check.ge(0.0)),
            "int_col": pa.Column(int),
            "string_col": pa.Column(str),
        }
    )
}

raise_exec_on_input = False
raise_exec_on_output = False

df = node1_pandas(
    input_df=pandas_df,
    input_validation=input_validation,
    output_validation=output_validation,
    raise_exec_on_input=raise_exec_on_input,
    raise_exec_on_output=raise_exec_on_output,
)
```

### Validating outputs returned as list/tuple/dictionary

#### Validating outputs returned as list/tuple
To validate the outputs returned as list/tuple, pass the output dataframes and schemas as dictionary where key is of type integer that refers to the index of the output dataframe in the returned list/tuple of dataframes and value is an object of DataFrameSchema class.

```{code-cell}
:tags: ["hide-cell"]

data = [
    {"id1": 101, "id2": "A_01", "name": "Adam"},
    {"id1": 102, "id2": "A_02", "name": "Smith"},
    {"id1": 102, "id2": "A_02", "name": "George"},
    {"id1": 104, "id2": "A_04", "name": "Jane"},
    {"id1": 105, "id2": None, "name": "John"},
]

pandas_df1 = pd.DataFrame(data)

data = [
    {"float_col": 1.0, "int_col": 1, "string_col": "foo",},
    {"float_col": 1.0, "int_col": 2, "string_col": "blabla",},
    {"float_col": 1.0, "int_col": 3, "string_col": None,},
]

pandas_df2 = pd.DataFrame(data)
```

```{code-cell}
print(pandas_df1)
```
```{code-cell}
print(pandas_df2)
```
```{code-cell}
@validate()
def node2_pandas(df1: pd.DataFrame, df2: pd.DataFrame):
    return df1.drop_duplicates("id1"), df2

input_validation = {
    "df1": pa.DataFrameSchema(
        {
            "id1": pa.Column(int, nullable=False), # composite primary key column
            "id2": pa.Column(str, nullable=False), # composite primary key column
            "name": pa.Column(str, nullable=True),
        },
        unique=["id1", "id2"], # checks joint uniqueness of composite primary key columns
    ),
    "df2": pa.DataFrameSchema(
        {".*_col": pa.Column(nullable=False, regex=True),}
    ),
}

output_validation = {
    0: pa.DataFrameSchema(
        {
            "id1": pa.Column(int),
            "id2": pa.Column(str),
            "name": pa.Column(str, nullable=True),
        },
        unique=["id1", "id2"],
    ),
    1: pa.DataFrameSchema({".*_col": pa.Column(nullable=True, regex=True)}),
}

out = node2_pandas(
    df1=pandas_df1,
    df2=pandas_df2,
    input_validation=input_validation,
    output_validation=output_validation,
)
```

#### Validating outputs returned as dictionary
To validate the outputs returned as dictionary, pass the output dataframes and schemas as dictionary where keys correspond to the keys of the returned dictionary and values are an object of DataFrameSchema class.

```{code-cell}
:tags: ["hide-cell"]

timestamp = datetime.datetime.fromtimestamp(time.time())

schema = StructType(
    [
        StructField("int_col", IntegerType(), True),
        StructField("long_col", LongType(), True),
        StructField("string_col", StringType(), True),
        StructField("float_col", FloatType(), True),
        StructField("double_col", DoubleType(), True),
        StructField("date_col", DateType(), True),
        StructField("datetime_col", TimestampType(), True),
        StructField("array_int", ArrayType(IntegerType()), True),
    ]
)

data = [
    (
        1,
        2,
        "awesome string",
        10.01,
        0.89,
        pd.Timestamp("2012-05-01").date(),
        timestamp,
        [1, 2, 3],
    ),
]

sample_df_spark_all_dtypes = spark.createDataFrame(data, schema)
```

```{code-cell}
spark_df.show()
```

```{code-cell}
@validate()
def node2_spark(input_df: pyspark.sql.DataFrame, input_df2: pyspark.sql.DataFrame):
    return {"out1": input_df, "out2": input_df2}

input_validation = {
    "input_df": py.DataFrameSchema(
        {
            "int_col": py.Column(T.IntegerType()),
            "float_col": py.Column(
                T.FloatType(), [py.Check.ge(0), py.Check.eq(90)]
            ),
            "string_col": py.Column(
                T.StringType(),
                [py.Check.str_startswith("."), py.Check.str_endswith("-")],
            ),
        },
    ),
    "input_df2": py.DataFrameSchema(
        {
            "int_col": py.Column(T.IntegerType()),
            "long_col": py.Column(T.LongType()),
            "string_col": py.Column(T.StringType()),
            "float_col": py.Column(T.FloatType()),
            "double_col": py.Column(T.DoubleType()),
            "date_col": py.Column(T.DateType()),
            "datetime_col": py.Column(T.TimestampType()),
            "array_int": py.Column(T.ArrayType(T.IntegerType())),
        }
    )
}
output_validation = {
    "out1": py.DataFrameSchema(
        {
            "int_col": py.Column(T.IntegerType()),
            "float_col": py.Column(T.FloatType()),
            "string_col": py.Column(T.StringType(), nullable=True),
        },
    ),
    "out2": py.DataFrameSchema(
        {
            "int_col": py.Column(T.IntegerType()),
            "long_col": py.Column(T.LongType()),
            "string_col": py.Column(T.StringType()),
            "float_col": py.Column(T.FloatType()),
            "double_col": py.Column(T.DoubleType()),
            "date_col": py.Column(T.DateType()),
            "datetime_col": py.Column(T.TimestampType()),
            "array_int": py.Column(T.ArrayType(T.IntegerType())),
        }
    ),
}

raise_exec_on_input = False
raise_exec_on_output = False


df = node2_spark(
    input_df=spark_df,
    input_df2=sample_df_spark_all_dtypes,
    input_validation=input_validation,
    output_validation=output_validation,
    raise_exec_on_input=raise_exec_on_input,
    raise_exec_on_output=raise_exec_on_output,
)
```

### Primary Key check
To check primary key, following constraints must be checked:
1. Key column must contain UNIQUE values
2. Key column cannot contain NULL values. [If null values are acceptable pass `nullable=True` for that column]

Note: Current version of Pandera doesn't support Unique constraint check for Pyspark dataframe. Therefore, primary key check can only be applied for Pandas datarames.

#### Single Column Primary Key
```{code-cell}
print(pandas_df2)
```

```{code-cell}
input_validation = {
    "input_df": pa.DataFrameSchema(
        {
            "float_col": pa.Column(float, nullable=True),
            "int_col": pa.Column(int,  unique=True, nullable=False), # primary key
            "string_col": pa.Column(str, nullable=True),
        }
    ),
}

out = node1_pandas(
    input_df=pandas_df2,
    input_validation=input_validation,
)
```

#### Composite Primary Key
```{code-cell}
print(pandas_df2)
```

```{code-cell}
input_validation = {
    "df1": pa.DataFrameSchema(
        {
            "id1": pa.Column(int, nullable=False),  # composite primary key
            "id2": pa.Column(str, nullable=False),  # composite primary key
            "name": pa.Column(str, nullable=True),
        },
        unique=["id1", "id2"],  # checks joint uniqueness
    ),
}

out = node2_pandas(
    df1=pandas_df1,
    df2=pandas_df2,
    input_validation=input_validation,
)
```

### Applying multiple data validations on a column
To apply multiple data validations on a column, pass them as a list in Column.

```{code-cell}
:tags: ["hide-cell"]
pandas_string_df = pd.DataFrame({"string_col": ["Asia", "Africa", "Europe"],})
```

```{code-cell}
print(pandas_string_df)
```

```{code-cell}
continents = ["Asia", "Africa", "Europe", "Antarctica"]
input_validation = {
    "input_df": pa.DataFrameSchema(
        {
            "string_col": pa.Column(
                str,
                [
                    pa.Check.str_matches(r"^[A-Z]"),
                    pa.Check.isin(continents),
                    pa.Check(lambda x: len(x) < 20),
                ],
            )
        }
    )
}

pd.DataFrame({"string_col": ["Asia", "Africa", "Europe"],})
out = node1_pandas(
    input_df=pandas_string_df, input_validation=input_validation
)
```
### Raising an exception in case of validation failure
By default, the pipelines do not fail in case of schema or data validation failure. 
To raise exception in case validation failure, use `raise_exec_on_output = True` for output mismatch and use `raise_exec_on_input = True` for input mismatch.

```{code-cell}

input_validation = {
    "input_df": pa.DataFrameSchema(
        {"string_col": pa.Column(str, pa.Check.str_matches(r"^[A-Z]"))}
    )
}

output_validation = {
    "out": pa.DataFrameSchema({"string_col": pa.Column(float)})
}

raise_exec_on_output = False
try:
    out = node1_pandas(
        input_df=pandas_string_df,
        input_validation=input_validation,
        output_validation=output_validation,
        raise_exec_on_output=raise_exec_on_output,
    )
except Exception:
    print("Exception raised on schema/data validation failure")
```