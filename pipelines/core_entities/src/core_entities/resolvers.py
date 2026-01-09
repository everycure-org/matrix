import os


def env(key: str, default: str = None, allow_null: str = False) -> str | None:
    """Load a variable from the environment.

    See https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html#custom-resolvers

    Args:
        key (str): Key to load.
        default (str): Default value to use instead
        allow_null (bool): Bool indicating whether null is allowed
    Returns:
        str: Value of the key
    """
    try:
        value = os.environ.get(key, default)
        if value is None and not allow_null:
            raise KeyError()
        return value
    except KeyError:
        raise KeyError(f"Environment variable '{key}' not found or default value {default} is None")
