from typing import Any
from inspect import isfunction
from functools import partial

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
        def wrapper(*args, **kwargs):
            print(foo)
            injected = inject(*args)
            return func(**injected)

        return wrapper

    return decorate


def train_model(estimator, scorer):
    print(estimator)
    print(scorer)


@inject_object(foo="baz")
def train_decorated(estimator, scorer):
    print(estimator)
    print(scorer)


train_decorated(
    {
        "estimator": {"_object": "xgboost.XGBClassifier"},
        "scorer": {"_object": "sklearn.metrics.f1_score", "average": "macro"},
    }
)
