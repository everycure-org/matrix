# ... new file ...

import os
import posixpath
import shutil
from typing import Any, Dict

import fsspec
from kedro.io.core import AbstractDataset, DatasetError


class LocalFileDataset(AbstractDataset[str, str]):
    """Save a local file path to a target filepath (e.g., GCS) using fsspec.
    - save(data): `data` must be a local filesystem path to an existing file.
    - load(): returns the target filepath URI (string).
    """

    def __init__(
        self,
        filepath: str,
        fs_args: Dict[str, Any] | None = None,
        credentials: Dict[str, Any] | None = None,
        save_args: Dict[str, Any] | None = None,
        load_args: Dict[str, Any] | None = None,
    ):
        self._filepath = filepath
        self._fs_args = fs_args or {}
        self._credentials = credentials or {}
        self._save_args = save_args or {}
        self._load_args = load_args or {}

        self._fs, self._path = fsspec.core.url_to_fs(filepath, **self._credentials, **self._fs_args)

    def _save(self, data: str) -> None:
        if not os.path.exists(data):
            raise DatasetError(f"Local file not found: {data}")

        parent = posixpath.dirname(self._path)
        if parent:
            try:
                self._fs.makedirs(parent, exist_ok=True)
            except Exception:
                pass

        chunk_size = int(self._save_args.get("chunk_size", 1024 * 1024))
        with open(data, "rb") as src, self._fs.open(self._path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=chunk_size)

    def _load(self) -> str:
        return self._filepath

    def _describe(self) -> Dict[str, Any]:
        return {"filepath": self._filepath}
