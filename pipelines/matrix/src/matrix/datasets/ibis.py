from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar

import ibis.expr.types as ir
from kedro.io import AbstractDataset, DatasetError

if TYPE_CHECKING:
    from ibis import BaseBackend


class TableDataset(AbstractDataset[ir.Table, ir.Table]):
    DEFAULT_LOAD_ARGS: ClassVar[dict[str, Any]] = {}
    DEFAULT_SAVE_ARGS: ClassVar[dict[str, Any]] = {
        "materialized": "view",
        "overwrite": True,
    }

    _connections: ClassVar[dict[tuple[tuple[str, str]], BaseBackend]] = {}

    def __init__(
        self,
        *,
        filepath: str | None = None,
        file_format: str | None = None,
        table_name: str | None = None,
        connection: dict[str, Any] | None = None,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
    ) -> None:
        if filepath is None and table_name is None:
            raise DatasetError(
                "Must provide at least one of `filepath` or `table_name`."
            )

        self._filepath = filepath
        self._file_format = file_format
        self._table_name = table_name
        self._connection_config = connection

        # Set load and save arguments, overwriting defaults if provided.
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)

        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

        self._materialized = self._save_args.pop("materialized")

    @property
    def connection(self) -> BaseBackend:
        cls = type(self)
        key = tuple(sorted(self._connection_config.items()))
        if key not in cls._connections:
            import ibis

            config = deepcopy(self._connection_config)
            backend = getattr(ibis, config.pop("backend"))
            cls._connections[key] = backend.connect(**config)

        return cls._connections[key]

    def _load(self) -> ir.Table:
        if self._filepath is not None:
            if self._file_format is None:
                raise NotImplementedError

            reader = getattr(self.connection, f"read_{self._file_format}")
            return reader(self._filepath, self._table_name, **self._load_args)
        else:
            return self.connection.table(self._table_name)

    def _save(self, data: ir.Table) -> None:
        if self._table_name is None:
            raise DatasetError("Must provide `table_name` for materialization.")

        writer = getattr(self.connection, f"create_{self._materialized}")
        writer(self._table_name, data, **self._save_args)

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath,
            "file_format": self._file_format,
            "table_name": self._table_name,
            "connection_config": self._connection_config,
            "load_args": self._load_args,
            "save_args": self._save_args,
            "materialized": self._materialized,
        }

    def _exists(self) -> bool:
        return (
            self._table_name is not None and self._table_name in self.connection.tables
        )
