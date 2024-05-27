# Pre-assignment: Dependency injection

## Context

Dependency injection is a programming technique that allows for decoupling object construction and usage. This usually boils down to functions receiving the objects that they rely on, as opposed to creating them internally. 

This technique is espcially relevant for data science workflows due to their widespread usage of objects, consider the following example:

```python
def train_model(
    data: pd.DataFrame, 
    features: List[str],
    target_col_name: str = "y"
) -> sklearn.base.BaseEstimator:
    
    # Initialise the classifier
    estimator = xgboost.XGBClassifier()
    
    # Index data
    mask = data["split"].eq("TRAIN")
    X_train = data.loc[mask, features]
    y_train = data.loc[mask, target_col_name]

    # Fit estimator
    return estimator.fit(X_train.values, y_train.values)
```

The code snippet above is tighly coupled to the `XGBClassifier`, while the code could easily be generalized to other models. Levaraging the dependency injection pattern would yield the following function definition:

```python
def train_model(
    data: pd.DataFrame, 
    features: List[str],
    estimator: sklearn.base.BaseEstimator,
    target_col_name: str = "y",
) -> sklearn.base.BaseEstimator:
    
    # Index data
    mask = data["split"].eq("TRAIN")
    X_train = data.loc[mask, features]
    y_train = data.loc[mask, target_col_name]

    # Fit estimator
    return estimator.fit(X_train.values, y_train.values)
```

## Assignment

The goal of this assignment is to implement the `inject` function:

- The function takes as an input the `func` to invoke, along with `params`, a definition of its dependencies. 

- The function should output the result of calling `func` with the correct dependencies as defined by `params`.

```python
def inject(func: Callable, params: Dict[str, Any]):
    ...
```

Dependencies are specified in the `params` dictionary using the following format following format:

```python
{
    "_object": "sklearn.base.BaseEstimator",
    "tree_method": "hist"
}
```

The `_object` key signals to the `inject` function that the object corresponding to its value should be injected, with the other `key:value` pairs as construction arguments.

Note that `params` may defined recusively, i.e., a `key:value` pair of the dictionary may contain another set of instructions for dependency injection, e.g.,

```
{
    "_object": "my.custom.ClassWithDependency",
    "dependency": {
        "_object": "my.custom.Dependency",
        "foo": "bar"
    }
}
```

In the example above, the `ClassWithDependency` will receive an instance of `Dependency` on construction.

## Follow-up questions

- How would you update your code to also allow injecting functions?

## Helping the candidate

- Ask them to verbally define their recusion strategy, i.e., what is the base case, and what is the recursive case?

- Point the candidate to [importlib](https://docs.python.org/3/library/importlib.html#importlib.import_module)