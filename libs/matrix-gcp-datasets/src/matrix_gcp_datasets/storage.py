import abc
import glob
import logging
import os
from pathlib import Path

from git import InvalidGitRepositoryError, NoSuchPathError, Repo


class StorageService(abc.ABC):
    """
    Abstract base class defining a storage service.
    """

    def __init__(self, root_path: str):
        self._root_path = root_path

    @abc.abstractmethod
    def exists(self, file_path: Path) -> bool:
        """
        Function to verify whether given path exists.

        Args:
            file_path: Path to the file
        Returns:
            boolean representing existence
        """

    @abc.abstractmethod
    def ls(self, glob_path: str) -> list[str]:
        """
        Function to list files in the given path.

        Args:
            glob_path: Path to the directory
        Returns:
            list of paths to files in the directory
        """

    @abc.abstractmethod
    def get(self, file_path: Path) -> str | None:
        """
        Function to retrieve the contents of the given path.

        Args:
            file_path: Path to the file
        Returns:
            string representing file contents
        """

    @abc.abstractmethod
    def save(self, file_path: Path | list[Path], contents: str, overwrite: bool = False, **kwargs) -> Path:
        """
        Function to save data in the given location.

        Args:
            file_path:  path or paths to files
            contents: file contents
            overwrite: boolean indicating file can be overwritten
        Returns:
            path to the materialized file
        """


class LocalStorageService(StorageService):
    """
    Specific StorageService that materializes files locally.
    """

    def exists(self, file_path: Path) -> bool:
        """
        Function to verify whether given path exists.

        Args:
            file_path: Path to the file
        Returns:
            boolean representing existence
        """
        full_path = Path(self._root_path) / file_path
        return full_path.exists()

    def ls(self, glob_path: str) -> list[Path]:
        """
        Function to list files in the given path.

        Args:
            glob_path: Path to the directory
        Returns:
            list of paths to files in the directory
        """
        globs = glob.glob(
            f"{self._root_path}/{glob_path}",
            recursive=True,
        )

        return [Path(glob).relative_to(self._root_path) for glob in globs]

    def get(self, file_path: Path) -> str | None:
        """
        Function to retrieve the contents of the given path.

        Args:
            file_path: Path to the file
        Returns:
            string representing file contents
        """
        full_path = Path(self._root_path) / file_path

        if full_path.exists():
            return full_path.open(encoding="utf-8").read()

        return None

    def save(self, file_path: Path, contents: str, overwrite: bool = False, **kwargs) -> Path:
        """
        Function to save data in the given location.

        Args:
            file_path: file destination path
            contents: file contents
            overwrite: boolean indicating file can be overwritten
        Returns:
            path to the materialized file
        """

        full_path = Path(self._root_path) / file_path

        if overwrite is False and full_path.exists():
            raise FileExistsError()

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with full_path.open("w+", encoding="utf-8") as file:
            file.write(contents)

        return full_path


class GitStorageService(LocalStorageService):
    _instance = None

    def __new__(cls, root_path: str, user: str, email: str, remote: str = "origin"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, root_path: str, user: str, email: str, remote: str = "origin"):
        # prevent reinitialization
        if getattr(self, "_initialized", False):
            return

        try:
            self._root_path = root_path
            self._remote = remote
            self._repo = Repo(str(root_path))

            with self._repo.config_writer() as git_config:
                git_config.set_value("user", "email", email)
                git_config.set_value("user", "name", user)

            logging.info(f"✅ Initialized GitStorageService at {root_path}")
        except (InvalidGitRepositoryError, NoSuchPathError) as err:
            logging.error(f"❌ Git repo error: {err}")
            raise err

        self._initialized = True

    def save(
        self,
        file_path: Path,
        contents: str,
        overwrite: bool = False,
        fetch_latest: bool = True,
        auto_commit: bool = True,
        commit_msg: str = None,
        **kwargs,
    ) -> Path:
        """
        Function to save data in the given location.

        Args:
            file_path:  path or paths to files
            contents: file contents
            fetch_latest: boolean indicating to fetch latest state before saving
            overwrite: boolean indicating file can be overwritten
            auto_commit: boolean indicating whether to auto-commit changes
            commit_msg: commit message
        Returns:
            path to the materialized file
        """
        if fetch_latest:
            self._repo.remote(self._remote).pull()

        full_path = super().save(file_path, contents, overwrite)

        if auto_commit and full_path is not None:
            commit_msg = f"add {os.path.basename(full_path)}" if not commit_msg else commit_msg
            self.commit_and_push([file_path], commit_msg)

        return full_path

    def commit_and_push(self, file_paths: list[Path], msg: str):
        """
        Helper function to commit and push the given file.
        """
        self._repo.git.add(file_paths)
        self._repo.index.commit(msg)
        push = self._repo.remote(self._remote).push()
        push.raise_if_error()

    @staticmethod
    def get_instance() -> "GitStorageService":
        if GitStorageService._instance is None:
            raise RuntimeError("GitStorageService has not been initialized yet.")
        return GitStorageService._instance
