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
```{code-cell}
:tags: ["hide-input"]
from refit.v1 import core
import pkgutil
from types import FunctionType


pkgname = core.__name__
pkgpath = core.__path__[0]
found_packages = list(pkgutil.iter_modules([pkgpath], prefix="{}.".format(pkgname)))
sub_packages = [x.split(".")[-1] for _, x, _ in found_packages]

importer = found_packages[0][0]


func_row = []
for idx, name in enumerate(sub_packages):
    module_spec = importer.find_spec(found_packages[idx][1])
    module = module_spec.loader.load_module(found_packages[idx][1])

    list_of_tags = [
        x
        for x in dir(module)
        if isinstance(getattr(module, x), FunctionType) and not x.startswith("_")
        and x not in ["getfullargspec", "wraps", "deepcopy"]
    ]

    from tabulate import tabulate

    for x in list_of_tags:
        x_doc = getattr(module, x).__doc__.split("\n")[0]
        func_row.append({"sub_module": module.__name__, "decorator": x, "description": x_doc})

import pandas as pd
table = pd.DataFrame(func_row)
print(tabulate(table, headers=table.columns, tablefmt="psql"))
```


Note that the decorators only work when extra keyword arguments are supplied to a
decorated function. This does modify the original function signature, and is
known as the adaptor pattern (https://en.wikipedia.org/wiki/Adapter_pattern). Note that
the decorators are mostly "off" by default and require a specific kwarg to be supplied
before being activated.
