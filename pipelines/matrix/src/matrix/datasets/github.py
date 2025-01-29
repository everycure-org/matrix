from typing import Any

import requests
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
        # ... get release API URL from repository URL
        release_url = repository_url.replace("github.com", "api.github.com/repos") + "/releases"

        # ... get releases' JSON
        requests_headers = {"Accept": "application/json"}
        if fs_args and "Authorization" in fs_args.keys():
            requests_headers["Authorization"] = fs_args["Authorization"]
        response = requests.get(release_url, headers=requests_headers)
        if response.status_code == 200:
            release_json = response.json()
        else:
            raise ValueError(f"Failed to get release JSON: {response.status_code} - {response.text}")

        # ... find the release
        release = next((r for r in release_json if r["name"] == release_name), None)
        if not release:
            raise ValueError(f"Could not find release {release_name} in releases {[r['name'] for r in release_json]}")

        # ... find the asset_id in release's assets
        asset_id = next((a["id"] for a in release["assets"] if a["name"] == release_asset_name), None)
        if not asset_id:
            raise ValueError(
                f"Could not find file {release_asset_name} in release {release_name} assets {[a['name'] for a in release['assets']]}"
            )

        asset_url = f"{release_url}/assets/{asset_id}"

        super().__init__(
            filepath=asset_url,
            fs_args=fs_args,
            load_args=load_args,
            metadata=metadata,
        )
