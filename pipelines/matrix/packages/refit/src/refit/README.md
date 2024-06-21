# Refit
Help keeps your code and pipeline lean by adapting your functions to be more pipeline
friendly.

How do you use `refit`? Let's say you have a normal function:
```
def fit_data(data: pd.DataFrame, X: List[str], y: str, model_object: BaseEstimator):
    fitted_model_object = model_object.fit(data[X], data[y])
    return fitted_model_object
```

You simply need to add a `refit` decorator (using the `@augment` decorator in this
example):
```python
from refit.v1.core.augment import augment


@augment()
def fit_data(data: pd.DataFrame, X: List[str], y: str, model_object: BaseEstimator):
    fitted_model_object = model_object.fit(data[X], data[y])
    return fitted_model_object
```

And you are good to go! Your function will instantly become more pipeline friendly.
More explanation below.


## Motivation and Existing Problems

A few useful tips to keep in mind when it comes to pipelining:

* All nodes are functions but not all functions are nodes.
* Nodes should do a useful piece of work - where a useful piece of work is defined as
an output worth persisting.
* Functions written as nodes should still be readable - containing the actual logic
and clearly documented.
* Running a function in any orchestration engine should not affect the way we write the function.

### Interactive Coding vs Pipelines

When writing a function meant to be use in an interactive session (such as a notebook
or terminal), it is not uncommon to write a function like below:
```
def fit_data(data: pd.DataFrame, X: List[str], y: str, model_object: BaseEstimator): -> BaseEstimator
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
```

However, when we see a similar function being used as part a pipeline or node, the following pattern
is more commonly seen:
```
def fit_data(data: pd.DataFrame, params: Dict[str, Any]) -> Base Estimator:
    """Fit a model on data.

    Args:
        data: A pandas dataframe.
        params: A dictionary containing X, y, and model object string. The dictionary
            format should be:
            ::
                {
                    "features": ["feature1", "feature2"]
                    "target": ["target_column"]
                    "model": "sklearn.linear_model.LinearRegression"
                }

    Returns:
        A fitted model object.
    """
    X = params.get("features")
    y = params.get("target")

    model_object = load_obj(params.get("model"))(**params.get("model_kwargs"))

    fitted_model_object = model_object.fit(data[X], data[y])
    return fitted_model_object
```

Notice a few things:

* Typehints are less clear in the pipeline version:
    - generic `Dict` or `Mapping` typehints.
    - Docstrings are less explicit. May contain an example dictionary that causes
        maintenance headaches in the long run.
* Pipeline version of the function has more code to `get` parameters out of dictionaries.
* More boiler plate code that dilutes the actual logic in the function.

The main reason is that when interacting with a function from a pipeline, often times
one only has access to YAML or some sort of equivalent, where as in an interactive
setting, one can instantiate objects before using them.

Using this decorator, we can keep our original function clean of pipeline related code,
keeping it lean and easier to review. The additional benefit is that every object
needed is instantiated in a standardised way.

Notice that in the "bad" example, the function receives a string and instantiates the
object within the function. This is discouraged according to the dependency injection
principle. For more comparison examples, see https://python-dependency-injector.ets-labs.org/introduction/di_in_python.html.

If we were working in an interactive setting, the following is the more normal
way to work with the function:
```
my_model = LinearRegressor()  # instantiate then use, more natural way

fitted_model = fit_data(data, X, y, my_model)
```


### Nodes should do a useful piece of work

A common pattern observed in pipelines where a filter operation is to be performed
because the subsequent function should not be run on the full data:
```
my_pipeline = Pipeline([
    Node(filter_func, "df_input", "df_filtered"),
    Node(actual_func, "df_filtered", "df_output"),
])
```
The implications:

* This means that `df_filtered` needs to added to the catalog. Expending unnecessary
mental energy to come up with a proper name and comply with any existing naming
conventions.
* Alternatively, one may choose to make `df_filtered` a `MemoryDataset` (in kedro), but
this leads to integration problems (i.e. converting a Kedro pipeline to `dataiku` or
`argo`).
* One may embed an optional line in `actual_func` to do filtering, but this will lead
to the `actual_func` being unncessarily longer.


`Refit` aims to help address the problems stated above by providing a set of
decorators to help reduce boilerplate code to make nodes more pipeline friendly.
Making your pipeline and underlying code more lean in the process, while also maintaining
the usability of functions in a non-pipeline scenario.

## Decorators

Below is a summary of the available decorators:



    +----+-------------------------------------------+-----------------------------+------------------------------------------------------------------------------------+
    |    | sub_module                                | decorator                   | description                                                                        |
    |----+-------------------------------------------+-----------------------------+------------------------------------------------------------------------------------|
    |  0 | refit.v1.core.augment                     | augment                     | Adds additional node functionalities to any function.                              |
    |  1 | refit.v1.core.columns_no_nulls            | columns_no_nulls            | Checks for nulls in specified columns.                                             |
    |  2 | refit.v1.core.fill_nulls                  | fill_nulls                  | Fills null values for output dataframe.                                            |
    |  3 | refit.v1.core.has_schema                  | has_schema                  | Checks the schema of ``input`` or ``output`` according to given ``schema``.        |
    |  4 | refit.v1.core.inject                      | inject_object               | Inject object decorator.                                                           |
    |  5 | refit.v1.core.inline_has_schema           | has_schema                  | Check if the input or output dataframe has the expected schema.                    |
    |  6 | refit.v1.core.inline_primary_key          | primary_key                 | Checks the primary key validation for list of columns of a dataframe.              |
    |  7 | refit.v1.core.inline_remove_debug_columns | remove_input_debug_columns  | Remove columns with a given prefix from all dataframe inputs configurable inline.  |
    |  8 | refit.v1.core.inline_remove_debug_columns | remove_output_debug_columns | Remove columns with a given prefix from all dataframe outputs configurable inline. |
    |  9 | refit.v1.core.input_kwarg_filter          | add_input_kwarg_filter      | Modifies function definition to include an additional filter kwarg at the input.   |
    | 10 | refit.v1.core.input_kwarg_select          | add_input_kwarg_select      | Modifies function definition to include an additional select kwarg at the input.   |
    | 11 | refit.v1.core.make_list_regexable         | make_list_regexable         | Allow processing of regex in input list.                                           |
    | 12 | refit.v1.core.output_filter               | add_output_filter           | Modifies function definition to include an additional filter kwarg at the output.  |
    | 13 | refit.v1.core.output_primary_key          | add_output_primary_key      | Primary key check function decorator.                                              |
    | 14 | refit.v1.core.pandera                     | validate                    | Validate the schema/data of `input` or `output` according to given schema.         |
    | 15 | refit.v1.core.primary_key                 | primary_key                 | Check the primary key constraints on dataframes.                                   |
    | 16 | refit.v1.core.remove_debug_columns        | remove_input_debug_columns  | Remove all columns with a given prefix from all dataframe inputs.                  |
    | 17 | refit.v1.core.remove_debug_columns        | remove_output_debug_columns | Remove all columns with a given prefix from all dataframe outputs.                 |
    | 18 | refit.v1.core.retry                       | retry                       | Retry function decorator.                                                          |
    | 19 | refit.v1.core.unpack                      | unpack_params               | Unpack params decorator.                                                           |
    +----+-------------------------------------------+-----------------------------+------------------------------------------------------------------------------------+

Note that the decorators only work when extra keyword arguments are supplied to a
decorated function. This does modify the original function signature, and is
known as the adaptor pattern (https://en.wikipedia.org/wiki/Adapter_pattern). Note that
the decorators are mostly "off" by default and require a specific kwarg to be supplied
before being activated.
