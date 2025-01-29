from typing import Any

import requests
from kedro.io.core import PROTOCOL_DELIMITER
from kedro_datasets.pandas import CSVDataset


class GitHubReleaseCSVDataset(CSVDataset):
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

        pwal = repository_url.replace("github.com", "api.github.com/repos") + "/releases"

        super().__init__(
            # ... we are not getting the actual file path in the init to avoid HTTP requests during tests
            filepath=pwal,
            fs_args=fs_args,
            load_args=load_args,
            metadata=metadata,
        )

    def load(self):
        # ... get release API URL from repository URL
        release_url = self.repository_url.replace("github.com", "api.github.com/repos") + "/releases"

        # ... get releases' JSON
        request_headers = {"Accept": "application/json"}
        if self.fs_args and "Authorization" in self.fs_args.keys():
            request_headers["Authorization"] = self.fs_args["Authorization"]
        response = requests.get(release_url, headers=request_headers)
        if response.status_code == 200:
            release_json = response.json()
        else:
            raise ValueError(f"Failed to get release JSON: {response.status_code} - {response.text}")

        # ... find the release
        release = next((r for r in release_json if r["name"] == self.release_name), None)
        if not release:
            raise ValueError(
                f"Could not find release {self.release_name} in releases {[r['name'] for r in release_json]}"
            )

        # ... find the asset_id in release's assets
        asset_id = next((a["id"] for a in release["assets"] if a["name"] == self.release_asset_name), None)
        if not asset_id:
            raise ValueError(
                f"Could not find file {self.release_asset_name} in release {self.release_name} assets {[a['name'] for a in release['assets']]}"
            )

        # ... CSVDataset load function adds the protocol and protocol delimiter to filepath, so we need to remove them
        # ... link to source code: https://github.com/kedro-org/kedro-plugins/blob/main/kedro-datasets/kedro_datasets/pandas/csv_dataset.py
        asset_url = f"{release_url}/assets/{asset_id}".removeprefix(f"{self._protocol}{PROTOCOL_DELIMITER}")
        self._filepath = asset_url

        return super().load()
