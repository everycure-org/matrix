import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from git import InvalidGitRepositoryError, Repo
from kedro.framework.hooks import hook_impl
from kedro.io.core import AbstractDataset, DatasetError
from kedro_datasets import pandas, spark
from pydantic import BaseModel, Field, field_serializer
from semantic_version import NpmSpec, Version

from matrix_gcp_datasets.storage import GitStorageService

logger = logging.getLogger(__name__)

app = typer.Typer()


class OwnerModel(BaseModel):
    name: str
    email: str | None = None


class LocationModel(BaseModel):
    type_: str = Field(alias="type")
    uri: str
    format_: str = Field(alias="format")


class DatasetModel(BaseModel):
    name: str
    version: str
    description: str | None = None
    message: str | None = None

    location: LocationModel
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    owner: OwnerModel
    tags: list[str] | None = None

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime, _info):
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def best_match(versions: list[str], pattern: str) -> tuple[str | None, bool]:
    """Function to find the best semver match.

    Args:
        versions: List of available versions
        pattern: semver pattern to match
    Returns:
        Best match, and boolean indicating whether this is the last version.
    """
    spec = NpmSpec(pattern)
    parsed_versions = [Version(v) for v in versions]

    # Find versions that satisfy the pattern
    matching = [v for v in parsed_versions if v in spec]
    if not matching:
        return None, False

    best_version = max(matching)
    latest_version = max(parsed_versions)
    is_latest = best_version == latest_version

    return str(best_version), is_latest


class GitStorageHook:
    """Kedro hook to clone or update a Git repository before running a pipeline."""

    def __init__(
        self,
        repo_url: str,
        target_dir: str = "data/external/data-catalog",
        branch: str = "main",
        force: bool = False,
        pull: bool = True,
    ):
        self.repo_url = repo_url
        self.target_dir = Path(target_dir)
        self.branch = branch
        self.force = force
        self.pull = pull

    @hook_impl
    def after_context_created(self, context):
        """Clone or update repo before the pipeline runs."""

        if self.force and self.target_dir.exists():
            shutil.rmtree(self.target_dir)

        if not self.target_dir.exists():
            repo = Repo.clone_from(self.repo_url, self.target_dir, branch=self.branch)

        else:
            try:
                repo = Repo(self.target_dir)
                logger.info(f"📁 Existing repo found: {repo.working_dir}")

                if self.pull:
                    logger.info(f"⬇️ Pulling latest changes from {self.branch}")
                    origin = repo.remotes.origin
                    origin.fetch()
                    origin.pull(self.branch)
                    logger.info("✅ Repository updated.")
                else:
                    logger.info("🔸 Skipping pull, using existing contents.")

                # Ensure branch consistency
                repo.git.checkout(self.branch)
            except InvalidGitRepositoryError:
                logger.info(f"⚠️ {self.target_dir} exists but is not a Git repo. Re-cloning.")
                shutil.rmtree(self.target_dir)
                Repo.clone_from(self.repo_url, self.target_dir, branch=self.branch)

        self.git_service = GitStorageService(
            root_path=str(self.target_dir),
            user="Kedro",
            email="kedro@everycure.org",
        )

        logger.info(f"🔁 GitStorageService ready at {self.target_dir}")


class DataCatalogDataset(AbstractDataset):
    """Custom dataset to load and read resources

    Examples:

        ```yaml
            catalog_diseases:
                type: matrix_gcp_datasets.catalog.DataCatalogDataset
                dataset: name

                # Used to define the engine to load the dataset
                # consider supporting polars, spark and pandas.
                engine: spark

                load_args:
                    # Version pattern to load, specified as npm version pattern
                    version: ~0.0.1

                    # Ensures a dataset error is thrown if a newer version is
                    # available that no longer matches the semver pattern
                    assert_latest: false

                save_args:
                    # Save args required to correcly materialize file using corresponding
                    # Kedro dataset, `{version}` pattern can be used to parametrize.
                    format: tsv
                    filepath: gs://mtrx-us-central1-hub-dev-storage/lvi/{version}/dataset.tsv

                    # Additional save arguments are passed directly
                    # to the respective dataset using the engine.
                    inferSchema: True
                    mode: overwrite
        ```
    """

    def __init__(
        self,
        *,
        dataset: str | dict[str, Any],
        engine: str,
        save_args: dict[str, Any] = None,
        load_args: dict[str, Any] = None,
        **kwargs,
    ):
        self._dataset = dataset
        self._engine = engine
        self._storage_service = GitStorageService.get_instance()
        self._save_args = save_args or {}
        self._load_args = load_args or {}

        self._assert_latest = self._load_args.pop("assert_latest", False)
        self._semvar_pattern = self._load_args.pop("version", "*")
        self._save_version = self._save_args.pop("version", None)
        self._message = self._save_args.pop("message", None)
        self._format = self._save_args.pop("format", "parquet")
        self._name = self._save_args.pop("name", None)
        self._email = self._save_args.pop("email", None)
        self._tags = self._save_args.pop("tags", None)
        self._metadata = self._save_args.pop("metadata", None)
        self._description = self._save_args.pop("description", None)
        self._filepath = self._save_args.pop("filepath", None)

    @property
    def filepath(self) -> str:
        version, _ = best_match(self.versions, self._semvar_pattern)
        dataset = DatasetModel.model_validate(
            yaml.safe_load(self._storage_service.get(Path(f"datasets/{self._dataset}/{version}/dataset.yaml")))
        )

        return dataset.location.uri

    @property
    def versions(self) -> list[str]:
        """Function to get versions for dataset."""
        paths = self._storage_service.ls(Path(f"datasets/{self._dataset}/*"))
        return [str(path.relative_to(Path(f"datasets/{self._dataset}"))) for path in paths]

    def load(self) -> Any:
        """Dataset loading

        Dataset loads the best matching version of the requested
        dataset using the pattern.
        """
        version, is_latest = best_match(self.versions, self._semvar_pattern)

        if version is None:
            raise DatasetError(
                f"No version matched for dataset '{self._dataset}', available versions: {','.join(self.versions)}"
            )

        if self._assert_latest and not is_latest:
            raise DatasetError(f"Newer version for dataset '{self._dataset}' available!")

        logger.info(f"Using version {version} for dataset '{self._dataset}'")
        try:
            dataset = DatasetModel.model_validate(
                yaml.safe_load(self._storage_service.get(Path(f"datasets/{self._dataset}/{version}/dataset.yaml")))
            )

            return self.get_dataset(
                dataset.location.format_, dataset.location.uri, self._load_args, self._save_args
            ).load()
        except Exception as e:
            raise DatasetError(f"Failed to load version for dataset '{self._dataset}': {e}") from e

    def get_dataset(
        self, format_: str, file_path: str, load_args: dict[str, Any], save_args: dict[str, Any]
    ) -> AbstractDataset:
        if self._engine == "spark":
            if format_ == "tsv":
                return spark.SparkDataset(
                    filepath=file_path,
                    file_format="csv",
                    load_args={**load_args, "sep": "\t", "header": True, "index": False},
                    save_args=save_args,
                )

            return spark.SparkDataset(
                filepath=file_path,
                file_format=format_,
                load_args={**load_args, "header": True, "index": False},
                save_args=save_args,
            )

        if self._engine == "pandas":
            if format_ == "csv":
                return pandas.CSVDataset(filepath=file_path, load_args=load_args, save_args=save_args)

            if format_ == "parquet":
                return pandas.ParquetDataset(filepath=file_path, load_args=load_args, save_args=save_args)

        raise ValueError(f"Unsupported engine: {(self._engine,)}")

    def get_schema(self, data) -> dict[str, str]:
        if self._engine == "pandas":
            type_map = {
                "int64": "int",
                "Int64": "int",
                "float64": "float",
                "object": "string",
                "bool": "bool",
                "datetime64[ns]": "datetime",
            }
            return {col: type_map.get(str(dtype), "unknown") for col, dtype in data.dtypes.items()}

        elif self._engine == "spark":
            spark_map = {
                "IntegerType()": "int",
                "LongType()": "int",
                "DoubleType()": "float",
                "FloatType()": "float",
                "StringType()": "string",
                "BooleanType()": "bool",
                "TimestampType()": "datetime",
                "DateType()": "date",
            }
            return {field.name: spark_map.get(str(field.dataType), "unknown") for field in data.schema.fields}

        else:
            raise ValueError(f"Unsupported engine: {self._engine}")

    def save(self, data: Any) -> None:
        """Dataset saving

        Dataset is saved using the next relevant semversion based
        on the save arguments.
        """

        if self._format is None or self._filepath is None:
            raise DatasetError(
                f"Insufficient `save_args` provided for storing to the catalog, expected {','.join(['filepath', 'format'])}"
            )

        if not self._save_version:
            self._save_version, self._message = self.prompt_version_bump()

        try:
            Version(self._save_version)
        except Exception:
            raise DatasetError(
                f"Invalid semantic version '{self._save_version}' — expected semver format like '1.0.0'."
            )

        self.get_dataset(
            self._format, self._filepath.format(version=self._save_version), self._load_args, self._save_args
        ).save(data)

        self._storage_service.save(
            f"datasets/{self._dataset}/{self._save_version}/dataset.yaml",
            yaml.dump(
                DatasetModel(
                    name=self._dataset,
                    version=self._save_version,
                    description=self._description,
                    message=self._message,
                    schema=self.get_schema(data),
                    metadata=self._metadata,
                    location=LocationModel(
                        type="gcs", uri=self._filepath.format(version=self._save_version), format=self._format
                    ),
                    owner=OwnerModel(name=self._name, email=self._email),
                    tags=self._tags,
                ).model_dump(by_alias=True)
            ),
            commit_msg=f"🤖 Create version {self._save_version} for '{self._dataset}'",
        )

    def prompt_version_bump(self) -> tuple[str, str | None]:
        """Prompt user for bumping information."""
        parsed = [Version(v) for v in self.versions]
        current_version = max([*parsed, Version("0.0.0")])
        typer.echo(f"Saving dataset: '{self._dataset}'")
        typer.echo(f"Current version: '{current_version}'")

        allowed = ["major", "minor", "patch"]
        bump_type = typer.prompt("Which part to bump? (major/minor/patch)").lower()
        while bump_type not in allowed:
            bump_type = typer.prompt("Invalid choice. Please choose major, minor, or patch").lower()

        new_version = {
            "major": Version(major=current_version.major + 1, minor=0, patch=0),
            "minor": Version(major=current_version.major, minor=current_version.minor + 1, patch=0),
            "patch": Version(major=current_version.major, minor=current_version.minor, patch=current_version.patch + 1),
        }[bump_type]

        if not typer.confirm(f"Do you want to save dataset '{self._dataset}' with version '{new_version}'?"):
            typer.echo("Save cancelled.")
            return

        message = typer.prompt("Optional message", default="", show_default=False)
        return str(new_version), message or None

    def _describe(self) -> dict[str, Any]:
        pass
