# NOTE: This file was partially generated using AI assistance.

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from huggingface_hub import (
    HfApi,
    hf_hub_download,
    login,
    snapshot_download,
    upload_file,
)
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from kedro.io.core import (
    AbstractDataset,
    DatasetError,
    Version,
)
from kedro_datasets.pandas import CSVDataset, ParquetDataset
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class HuggingFaceDatasetError(DatasetError):
    """Base exception for HuggingFace dataset operations."""


class AuthenticationError(HuggingFaceDatasetError):
    """Raised when HuggingFace authentication fails."""


class HuggingFaceRepositoryNotFoundError(HuggingFaceDatasetError):
    """Raised when HuggingFace repository is not accessible."""


class HuggingFaceFileNotFoundError(HuggingFaceDatasetError):
    """Raised when file is not found in HuggingFace repository."""


class HuggingFaceBaseDataset(AbstractDataset, ABC):
    """Base dataset for HuggingFace Hub integration.
    
    This abstract base class provides common functionality for all HuggingFace
    dataset implementations, including authentication, file operations, and
    error handling.
    """

    def __init__(
        self,
        repo_id: str,
        filename: str,
        revision: str = "main",
        credentials: Optional[Dict[str, Any]] = None,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[Version] = None,
    ):
        """Initialize HuggingFace dataset.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "everycure/matrix-kg")
            filename: File name within the repository
            revision: Git revision (branch, tag, or commit hash)
            credentials: Authentication credentials containing 'token'
            load_args: Arguments for loading data
            save_args: Arguments for saving data including commit info
            metadata: Dataset metadata for repository
            version: Kedro dataset version
        """
        self._repo_id = repo_id
        self._filename = filename
        self._revision = revision
        self._credentials = credentials or {}
        self._load_args = load_args or {}
        self._save_args = save_args or {}
        self._metadata = metadata or {}
        self._version = version
        
        # Initialize HuggingFace API client
        self._api = HfApi()
        self._token = self._authenticate()

    def _authenticate(self) -> Optional[str]:
        """Handle HuggingFace authentication.
        
        Returns:
            Authentication token if available
            
        Raises:
            AuthenticationError: If authentication fails
        """
        token = None
        
        # Try to get token from credentials
        if "token" in self._credentials:
            token = self._credentials["token"]
        elif "token_file" in self._credentials:
            token_file = Path(self._credentials["token_file"]).expanduser()
            if token_file.exists():
                token = token_file.read_text().strip()
        else:
            # Try environment variable
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        if token:
            try:
                login(token=token)
                logger.info("Successfully authenticated with HuggingFace Hub")
                return token
            except Exception as e:
                raise AuthenticationError(f"Failed to authenticate with HuggingFace Hub: {e}")
        
        logger.warning("No HuggingFace token provided - using anonymous access")
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, HfHubHTTPError))
    )
    def _download_file(self) -> str:
        """Download file from HuggingFace Hub with retry logic.
        
        Returns:
            Path to downloaded file in local cache
            
        Raises:
            HuggingFaceRepositoryNotFoundError: If repository not found
            HuggingFaceFileNotFoundError: If file not found in repository
            HuggingFaceDatasetError: For other download errors
        """
        try:
            logger.info(f"Downloading {self._filename} from {self._repo_id}@{self._revision}")
            
            file_path = hf_hub_download(
                repo_id=self._repo_id,
                filename=self._filename,
                revision=self._revision,
                token=self._token,
                **self._load_args.get("download_args", {})
            )
            
            logger.info(f"Successfully downloaded file to {file_path}")
            return file_path
            
        except RepositoryNotFoundError as e:
            raise HuggingFaceRepositoryNotFoundError(
                f"Repository {self._repo_id} not found or not accessible: {e}"
            )
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise HuggingFaceFileNotFoundError(
                    f"File {self._filename} not found in repository {self._repo_id}: {e}"
                )
            else:
                raise HuggingFaceDatasetError(f"HTTP error downloading file: {e}")
        except Exception as e:
            raise HuggingFaceDatasetError(f"Failed to download file: {e}")

    def _upload_file(self, local_path: str) -> None:
        """Upload file to HuggingFace Hub.
        
        Args:
            local_path: Path to local file to upload
            
        Raises:
            HuggingFaceDatasetError: If upload fails
        """
        try:
            commit_message = self._save_args.get("commit_message", f"Update {self._filename}")
            commit_description = self._save_args.get("commit_description", "")
            
            logger.info(f"Uploading {local_path} to {self._repo_id}/{self._filename}")
            
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=self._filename,
                repo_id=self._repo_id,
                token=self._token,
                commit_message=commit_message,
                commit_description=commit_description,
                revision=self._revision,
                **self._save_args.get("upload_args", {})
            )
            
            logger.info(f"Successfully uploaded file to {self._repo_id}/{self._filename}")
            
        except Exception as e:
            raise HuggingFaceDatasetError(f"Failed to upload file: {e}")

    def exists(self) -> bool:
        """Check if dataset exists in HuggingFace Hub.
        
        Returns:
            True if file exists in repository, False otherwise
        """
        try:
            self._api.hf_hub_download(
                repo_id=self._repo_id,
                filename=self._filename,
                revision=self._revision,
                token=self._token,
            )
            return True
        except (RepositoryNotFoundError, HfHubHTTPError):
            return False

    def _describe(self) -> Dict[str, Any]:
        """Return dataset description for Kedro catalog.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            "repo_id": self._repo_id,
            "filename": self._filename,
            "revision": self._revision,
            "type": self.__class__.__name__,
        }

    @abstractmethod
    def _load(self) -> Any:
        """Load data from the downloaded file.
        
        This method should be implemented by concrete dataset classes
        to handle specific file formats.
        """

    @abstractmethod
    def _save(self, data: Any) -> None:
        """Save data to a local file for upload.
        
        This method should be implemented by concrete dataset classes
        to handle specific file formats.
        
        Args:
            data: Data to save
        """

    def load(self) -> Any:
        """Load data from HuggingFace Hub.
        
        Returns:
            Loaded data in appropriate format
        """
        file_path = self._download_file()
        return self._load_from_path(file_path)

    def save(self, data: Any) -> None:
        """Save data to HuggingFace Hub.
        
        Args:
            data: Data to save to repository
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{self._get_file_extension()}") as tmp_file:
            try:
                self._save_to_path(data, tmp_file.name)
                self._upload_file(tmp_file.name)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)

    @abstractmethod
    def _load_from_path(self, file_path: str) -> Any:
        """Load data from local file path."""

    @abstractmethod
    def _save_to_path(self, data: Any, file_path: str) -> None:
        """Save data to local file path."""

    @abstractmethod
    def _get_file_extension(self) -> str:
        """Get file extension for temporary files."""


class HuggingFaceParquetDataset(HuggingFaceBaseDataset):
    """Parquet format dataset for HuggingFace Hub.
    
    This dataset handles Parquet files stored in HuggingFace repositories,
    providing efficient storage and loading of structured data.
    """

    def _load_from_path(self, file_path: str) -> pd.DataFrame:
        """Load Parquet data from local file path.
        
        Args:
            file_path: Path to local Parquet file
            
        Returns:
            Loaded DataFrame
        """
        return pd.read_parquet(file_path, **self._load_args.get("pandas_args", {}))

    def _save_to_path(self, data: pd.DataFrame, file_path: str) -> None:
        """Save DataFrame to Parquet file.
        
        Args:
            data: DataFrame to save
            file_path: Path to save Parquet file
        """
        data.to_parquet(file_path, **self._save_args.get("pandas_args", {}))

    def _get_file_extension(self) -> str:
        """Get file extension for Parquet files."""
        return "parquet"


class HuggingFaceCSVDataset(HuggingFaceBaseDataset):
    """CSV format dataset for HuggingFace Hub.
    
    This dataset handles CSV files stored in HuggingFace repositories,
    providing compatibility with tabular data formats.
    """

    def _load_from_path(self, file_path: str) -> pd.DataFrame:
        """Load CSV data from local file path.
        
        Args:
            file_path: Path to local CSV file
            
        Returns:
            Loaded DataFrame
        """
        return pd.read_csv(file_path, **self._load_args.get("pandas_args", {}))

    def _save_to_path(self, data: pd.DataFrame, file_path: str) -> None:
        """Save DataFrame to CSV file.
        
        Args:
            data: DataFrame to save
            file_path: Path to save CSV file
        """
        data.to_csv(file_path, index=False, **self._save_args.get("pandas_args", {}))

    def _get_file_extension(self) -> str:
        """Get file extension for CSV files."""
        return "csv"


class HuggingFaceJSONDataset(HuggingFaceBaseDataset):
    """JSON format dataset for HuggingFace Hub.
    
    This dataset handles JSON files stored in HuggingFace repositories,
    providing support for semi-structured data formats.
    """

    def _load_from_path(self, file_path: str) -> Union[Dict, list]:
        """Load JSON data from local file path.
        
        Args:
            file_path: Path to local JSON file
            
        Returns:
            Loaded JSON data (dict or list)
        """
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f, **self._load_args.get("json_args", {}))

    def _save_to_path(self, data: Union[Dict, list], file_path: str) -> None:
        """Save data to JSON file.
        
        Args:
            data: Data to save (dict or list)
            file_path: Path to save JSON file
        """
        import json
        
        json_args = self._save_args.get("json_args", {})
        json_args.setdefault("indent", 2)
        json_args.setdefault("ensure_ascii", False)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, **json_args)

    def _get_file_extension(self) -> str:
        """Get file extension for JSON files."""
        return "json"


class HuggingFaceXetDataset(HuggingFaceBaseDataset):
    """Xet format dataset for large files on HuggingFace Hub.
    
    This dataset handles large files using the Xet file system format,
    providing efficient storage and transfer of large datasets.
    
    Note: This is a placeholder implementation. Full Xet integration
    would require the Xet Python SDK when it becomes available.
    """

    def __init__(self, *args, **kwargs):
        """Initialize Xet dataset with additional Xet-specific configuration."""
        self._xet_args = kwargs.pop("xet_args", {})
        super().__init__(*args, **kwargs)
        
        # Configure Xet settings
        self._chunk_size = self._xet_args.get("chunk_size", "64MB")
        self._compression = self._xet_args.get("compression", "lz4")
        self._lazy_loading = self._load_args.get("lazy", False)

    def _configure_xet(self) -> None:
        """Configure Xet file system settings.
        
        This method would configure Xet-specific settings when the
        Xet Python SDK becomes available.
        """
        logger.warning("Xet integration not yet implemented - falling back to standard download")

    def _load_from_path(self, file_path: str) -> Any:
        """Load data from Xet file.
        
        Args:
            file_path: Path to local Xet file
            
        Returns:
            Loaded data (format depends on file content)
        """
        # Placeholder implementation - would use Xet SDK
        # For now, attempt to load as Parquet (common for large datasets)
        try:
            return pd.read_parquet(file_path)
        except Exception:
            # Fallback to binary read for unknown formats
            with open(file_path, 'rb') as f:
                return f.read()

    def _save_to_path(self, data: Any, file_path: str) -> None:
        """Save data to Xet file.
        
        Args:
            data: Data to save
            file_path: Path to save Xet file
        """
        # Placeholder implementation - would use Xet SDK
        if isinstance(data, pd.DataFrame):
            data.to_parquet(file_path)
        elif isinstance(data, bytes):
            with open(file_path, 'wb') as f:
                f.write(data)
        else:
            raise ValueError(f"Unsupported data type for Xet dataset: {type(data)}")

    def _get_file_extension(self) -> str:
        """Get file extension for Xet files."""
        return "xet"

    def _upload_xet_file(self, local_path: str) -> None:
        """Upload large file using Xet format.
        
        This method would use Xet-specific upload mechanisms
        when the Xet Python SDK becomes available.
        
        Args:
            local_path: Path to local file to upload
        """
        logger.warning("Xet-specific upload not yet implemented - using standard upload")
        self._upload_file(local_path)

    def _download_xet_file(self) -> str:
        """Download large file from Xet storage.
        
        This method would use Xet-specific download mechanisms
        when the Xet Python SDK becomes available.
        
        Returns:
            Path to downloaded file
        """
        logger.warning("Xet-specific download not yet implemented - using standard download")
        return self._download_file()


# Convenience aliases for backward compatibility and easier imports
HuggingFaceDataset = HuggingFaceParquetDataset  # Default to Parquet format