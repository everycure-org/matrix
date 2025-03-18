from abc import ABC, abstractmethod
from typing import Any

import requests
from kedro.io.core import PROTOCOL_DELIMITER
from kedro_datasets.pandas import CSVDataset, ExcelDataset


class GitHubReleaseBaseDataset(ABC):
    def __init__(
        self,
        repository_url: str,
        release_name: str,
        release_asset_name: str,
        load_args: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.repository_url = repository_url
        self.release_name = release_name
        self.release_asset_name = release_asset_name
        self.load_args = load_args
        self.fs_args = fs_args
        self.metadata = metadata

        # get release API URL from repository URL
        self.release_url = self.repository_url.replace("github.com", "api.github.com/repos") + "/releases"

    def _get_release_json(self, release_url: str, fs_args: dict[str, Any] | None) -> dict[str, Any]:
        """
        Get the release's JSON from the GitHub API.
        """

        request_headers = {"Accept": "application/json"}
        if fs_args and "Authorization" in fs_args.keys():
            request_headers["Authorization"] = fs_args["Authorization"]
        response = requests.get(release_url, headers=request_headers)
        if response.status_code == 200:
            release_json = response.json()
        else:
            raise ValueError(f"Failed to get release JSON: {response.status_code} - {response.text}")

        return release_json

    def _get_asset_id(self, release_json: dict[str, Any], release_name: str, release_asset_name: str) -> str:
        """
        Get the asset ID from the release's JSON.
        """

        # find the release
        release = next((r for r in release_json if r["name"] == release_name), None)
        if not release:
            raise ValueError(f"Could not find release {release_name} in releases {[r['name'] for r in release_json]}")

        # find the asset_id in release's assets
        asset_id = next((a["id"] for a in release["assets"] if a["name"] == release_asset_name), None)
        if not asset_id:
            raise ValueError(
                f"Could not find file {release_asset_name} in release {release_name} assets {[a['name'] for a in release['assets']]}"
            )

        return asset_id

    @abstractmethod
    def load(self):
        pass


class GitHubReleaseCSVDataset(GitHubReleaseBaseDataset, CSVDataset):
    def __init__(
        self,
        repository_url: str,
        release_name: str,
        release_asset_name: str,
        load_args: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):  # Initialize base GitHub functionality
        GitHubReleaseBaseDataset.__init__(
            self,
            repository_url=repository_url,
            release_name=release_name,
            release_asset_name=release_asset_name,
            load_args=load_args,
            fs_args=fs_args,
            metadata=metadata,
        )
        # Initialize CSV dataset functionality
        CSVDataset.__init__(
            self,
            filepath=self.release_url,
            fs_args=fs_args,
            load_args=load_args,
            metadata=metadata,
        )

    def load(self):
        release_json = self._get_release_json(self.release_url, self.fs_args)
        asset_id = self._get_asset_id(release_json, self.release_name, self.release_asset_name)

        # CSVDataset load function adds the protocol and protocol delimiter to filepath, so we need to remove them
        # link to source code: https://github.com/kedro-org/kedro-plugins/blob/main/kedro-datasets/kedro_datasets/pandas/csv_dataset.py
        asset_url = f"{self.release_url}/assets/{asset_id}".removeprefix(f"{self._protocol}{PROTOCOL_DELIMITER}")
        self._filepath = asset_url

        return CSVDataset.load(self)


class GitHubReleaseExcelDataset(GitHubReleaseBaseDataset, ExcelDataset):
    def __init__(
        self,
        repository_url: str,
        release_name: str,
        release_asset_name: str,
        load_args: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        GitHubReleaseBaseDataset.__init__(
            self,
            repository_url=repository_url,
            release_name=release_name,
            release_asset_name=release_asset_name,
            load_args=load_args,
            fs_args=fs_args,
            metadata=metadata,
        )
        # Initialize Excel dataset functionality
        ExcelDataset.__init__(
            self,
            filepath=self.release_url,
            fs_args=fs_args,
            load_args=load_args,
            metadata=metadata,
        )

    def load(self):
        release_json = self._get_release_json(self.release_url, self.fs_args)
        asset_id = self._get_asset_id(release_json, self.release_name, self.release_asset_name)

        asset_url = f"{self.release_url}/assets/{asset_id}".removeprefix(f"{self._protocol}{PROTOCOL_DELIMITER}")
        self._filepath = asset_url

        return ExcelDataset.load(self)
