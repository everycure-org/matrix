import os
import tempfile
from typing import Any

import requests
from kedro_datasets.pandas import CSVDataset


class GitHubReleaseCSVDataset(CSVDataset):
    """
    A Kedro dataset for loading CSV files from GitHub release assets.

    This dataset downloads CSV files directly from GitHub release assets,
    handling authentication and temporary file management automatically.

    Example:
        ```python
        dataset = GitHubReleaseCSVDataset(
            repository_url="https://github.com/owner/repo",
            release_name="v1.0.0",
            release_asset_name="data.csv",
            fs_args={"Authorization": "token ghp_..."}
        )
        df = dataset.load()
        ```
    """

    def __init__(
        self,
        repository_url: str,
        release_name: str,
        release_asset_name: str,
        load_args: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the GitHub Release CSV Dataset.

        Args:
            repository_url: GitHub repository URL (e.g., "https://github.com/owner/repo")
            release_name: Name of the release (e.g., "v1.0.0")
            release_asset_name: Name of the CSV file in the release assets
            load_args: Additional arguments passed to pandas.read_csv()
            fs_args: File system arguments, including "Authorization" for GitHub token
            metadata: Arbitrary metadata for the dataset
        """
        self.repository_url = repository_url
        self.release_name = release_name
        self.release_asset_name = release_asset_name
        self.fs_args = fs_args

        # Convert repository URL to GitHub API URL
        self.release_url = self._build_api_url(repository_url)

        super().__init__(
            filepath=self.release_url,  # Placeholder filepath for parent class
            fs_args=fs_args,
            load_args=load_args,
            metadata=metadata,
        )

    def _build_api_url(self, repository_url: str) -> str:
        """Convert GitHub repository URL to API releases endpoint."""
        return repository_url.replace("github.com", "api.github.com/repos") + "/releases"

    def _get_release_json(self, release_url: str, fs_args: dict[str, Any] | None) -> list[dict[str, Any]]:
        """
        Fetch release data from GitHub API.

        Args:
            release_url: GitHub API releases endpoint URL
            fs_args: File system arguments containing authentication

        Returns:
            List of release objects from GitHub API

        Raises:
            ValueError: If the API request fails
        """
        headers = self._build_request_headers(fs_args, accept_type="application/json")

        response = requests.get(release_url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"Failed to get release JSON: {response.status_code} - {response.text}")

    def _build_request_headers(self, fs_args: dict[str, Any] | None, accept_type: str) -> dict[str, str]:
        """
        Build request headers including authentication if provided.

        Args:
            fs_args: File system arguments that may contain Authorization
            accept_type: The Accept header value for the request

        Returns:
            Dictionary of request headers
        """
        headers = {"Accept": accept_type}
        if fs_args and "Authorization" in fs_args:
            headers["Authorization"] = fs_args["Authorization"]
        return headers

    def _get_asset_id(self, releases: list[dict[str, Any]], release_name: str, release_asset_name: str) -> str:
        """
        Extract the asset ID for a specific file from release data.

        Args:
            releases: List of release objects from GitHub API
            release_name: Name of the target release
            release_asset_name: Name of the target asset file

        Returns:
            GitHub asset ID for the specified file

        Raises:
            ValueError: If release or asset is not found
        """
        # Find the target release
        release = next((r for r in releases if r["name"] == release_name), None)
        if not release:
            available_releases = [r["name"] for r in releases]
            raise ValueError(f"Could not find release '{release_name}' in releases {available_releases}")

        # Find the target asset within the release
        asset_id = next((a["id"] for a in release["assets"] if a["name"] == release_asset_name), None)
        if not asset_id:
            available_assets = [a["name"] for a in release["assets"]]
            raise ValueError(
                f"Could not find file '{release_asset_name}' in release '{release_name}' " f"assets {available_assets}"
            )

        return str(asset_id)

    def _download_asset_content(self, asset_url: str) -> str:
        """
        Download the content of a GitHub release asset.

        Args:
            asset_url: GitHub API URL for the asset download

        Returns:
            The content of the asset as a string

        Raises:
            ValueError: If the download fails
        """
        headers = self._build_request_headers(self.fs_args, accept_type="application/octet-stream")

        response = requests.get(asset_url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            raise ValueError(f"Failed to download asset: {response.status_code} - {response.text}")

    def _create_temp_file(self, content: str) -> str:
        """
        Create a temporary file with the downloaded content.

        Args:
            content: The CSV content to write to the file

        Returns:
            Path to the created temporary file
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
            temp_file.write(content)
            return temp_file.name

    def load(self):
        """
        Load CSV data from a GitHub release asset.

        Returns:
            pandas.DataFrame: The loaded CSV data
        """
        # Get release metadata from GitHub API
        releases = self._get_release_json(self.release_url, self.fs_args)

        # Find the specific asset we want
        asset_id = self._get_asset_id(releases, self.release_name, self.release_asset_name)

        # Build download URL for the asset
        asset_url = f"{self.release_url}/assets/{asset_id}"

        # Download the asset content
        content = self._download_asset_content(asset_url)

        # Create temporary file and use parent class to load it
        temp_filepath = self._create_temp_file(content)

        try:
            # Update filepath to point to temp file and use parent class
            original_filepath = self._filepath
            original_protocol = self._protocol

            self._filepath = temp_filepath
            self._protocol = "file"

            return super().load()

        finally:
            # Restore original values and clean up temp file
            self._filepath = original_filepath
            self._protocol = original_protocol

            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
