"""matrix-validator package."""

import importlib_metadata

try:
    __version__ = importlib_metadata.version(__name__)
except importlib_metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"  # pragma: no cover


def serialize_sets(obj):
    """Serialize sets for json."""
    if isinstance(obj, set):
        return list(obj)
    return obj
