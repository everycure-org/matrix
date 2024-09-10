from kedro.framework.session import KedroSession
from kedro.framework.context import KedroContext
from kedro.io.data_catalog import DataCatalog
from pathlib import Path
from kedro.framework.project import (
    pipelines,
    settings,
    validate_settings,
)


class CustomKedroContext(KedroContext):
    _catalog = None

    def _get_catalog(
        self,
        save_version: str | None = None,
        load_versions: dict[str, str] | None = None,
    ) -> DataCatalog:
        if self._catalog:
            return self._catalog

        print("initializing catalog")
        self._catalog = super()._get_catalog(save_version, load_versions)
        return self._catalog


class CustomKedroSession(KedroSession):
    def __init__(
        self,
        session_id: str,
        package_name: str | None = None,
        project_path: Path | str | None = None,
        save_on_close: bool = False,
        conf_source: str | None = None,
    ):
        self._context = None
        super().__init__(
            session_id, package_name, project_path, save_on_close, conf_source
        )

        self.load_context()

    def load_context(self) -> KedroContext:
        if self._context:
            return self._context

        print("initializing context")
        self._context = super().load_context()
        self._context._get_catalog()
        return self._context
