from typing import Any
from inspect import isfunction
from functools import partial
from functools import wraps

import pandas as pd

import importlib

_INJECT_PREFIX = "_object"


def import_(path: str, **kwargs) -> Any:
    module, obj = path.rsplit(".", 1)
    module = importlib.import_module(module)
    attr = getattr(module, obj)

    if isfunction(attr):
        return partial(attr, **kwargs)

    return attr(**kwargs)


def inject(config) -> Any:
    if isinstance(config, list):
        return [inject(el) for el in config]

    if isinstance(config, dict):
        path = config.pop(_INJECT_PREFIX, None)

        if path:
            return import_(path, **{k: inject(v) for k, v in config.items()})
        else:
            return {k: inject(v) for k, v in config.items()}

    return config


def inject_object(foo: str = "bar"):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # breakpoint()
            return func(
                *[inject(arg) for arg in args],
                **{k: inject(v) for k, v in kwargs.items()},
            )

        return wrapper

    return decorate


def train_model(df: pd.DataFrame, estimator, scorer):
    print(df)
    print(estimator)
    print(scorer)


@inject_object(foo="baz")
def train_decorated(data, estimator, scorer, test):
    print(data)
    print(estimator)
    print(scorer)
    print(test)


# train_model(
#     pd.DataFrame([["a", "b"]], columns=["foo", "bar"]),
#     **inject(
#         {
#             "estimator": {"_object": "xgboost.XGBClassifier"},
#             "scorer": {"_object": "sklearn.metrics.f1_score", "average": "macro"},
#         }
#     ),
# )

train_decorated(
    data=pd.DataFrame([["a", "b"]], columns=["foo", "bar"]),
    estimator={"_object": "xgboost.XGBClassifier"},
    scorer={"_object": "sklearn.metrics.f1_score", "average": "macro"},
    test={"_object": "sklearn.metrics.accuracy_score"},
)

train_decorated(
    pd.DataFrame([["a", "b"]], columns=["foo", "bar"]),
    {"_object": "xgboost.XGBClassifier"},
    {"_object": "sklearn.metrics.f1_score", "average": "macro"},
    {"_object": "sklearn.metrics.accuracy_score"},
)
