import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from matrix.utils.kubernetes import (
    can_talk_to_kubernetes,
    get_gcp_project_from_config,
    get_gcp_project_from_metadata,
    get_runtime_gcp_bucket,
    get_runtime_gcp_project_id,
    get_runtime_mlflow_url,
)


class TestGCPProjectAutoDetection:
    """Test suite for GCP project auto-detection functionality."""

    @patch("subprocess.check_output")
    def test_get_gcp_project_from_config_success(self, mock_subprocess):
        """Test successful project ID retrieval from gcloud config."""
        mock_subprocess.return_value = "mtrx-hub-dev-3of\n"

        result = get_gcp_project_from_config()

        assert result == "mtrx-hub-dev-3of"
        mock_subprocess.assert_called_once()

    @patch("subprocess.check_output")
    def test_get_gcp_project_from_config_unset(self, mock_subprocess):
        """Test handling of unset project in gcloud config."""
        mock_subprocess.return_value = "(unset)\n"

        with pytest.raises(RuntimeError, match="No active GCP project found"):
            get_gcp_project_from_config()

    @patch("subprocess.check_output")
    def test_get_gcp_project_from_config_gcloud_not_installed(self, mock_subprocess):
        """Test handling when gcloud is not installed."""
        mock_subprocess.side_effect = FileNotFoundError()

        with pytest.raises(EnvironmentError, match="gcloud is not installed"):
            get_gcp_project_from_config()

    @patch("subprocess.check_output")
    def test_get_gcp_project_with_impersonation(self, mock_subprocess):
        """Test project ID retrieval with service account impersonation."""
        mock_subprocess.return_value = "mtrx-hub-prod-sms\n"

        with patch.dict(os.environ, {"SPARK_IMPERSONATION_SERVICE_ACCOUNT": "test@example.com"}):
            result = get_gcp_project_from_config()

        assert result == "mtrx-hub-prod-sms"
        # Verify the impersonation flag was added
        call_args = mock_subprocess.call_args[0][0]
        assert "--impersonate-service-account=test@example.com" in call_args

    @patch("matrix.utils.kubernetes.get_gcp_project_from_metadata")
    @patch("matrix.utils.kubernetes.get_gcp_project_from_config")
    def test_get_runtime_gcp_project_id_metadata_success(self, mock_gcloud, mock_metadata):
        """Test that runtime project ID uses metadata server when available."""
        mock_metadata.return_value = "metadata-project"

        result = get_runtime_gcp_project_id()

        assert result == "metadata-project"
        mock_metadata.assert_called_once()
        mock_gcloud.assert_not_called()

    @patch("matrix.utils.kubernetes.get_gcp_project_from_metadata")
    @patch("matrix.utils.kubernetes.get_gcp_project_from_config")
    def test_get_runtime_gcp_project_id_gcloud_fallback(self, mock_gcloud, mock_metadata):
        """Test that runtime project ID falls back to gcloud when metadata fails."""
        mock_metadata.side_effect = RuntimeError("metadata server unavailable")
        mock_gcloud.return_value = "gcloud-project"

        result = get_runtime_gcp_project_id()

        assert result == "gcloud-project"
        mock_metadata.assert_called_once()
        mock_gcloud.assert_called_once()

    @patch("matrix.utils.kubernetes.get_gcp_project_from_metadata")
    @patch("matrix.utils.kubernetes.get_gcp_project_from_config")
    def test_get_runtime_gcp_project_id_all_fail(self, mock_gcloud, mock_metadata):
        """Test that runtime project ID fails when both metadata and gcloud fail."""
        mock_metadata.side_effect = RuntimeError("metadata server unavailable")
        mock_gcloud.side_effect = RuntimeError("gcloud config failed")

        with pytest.raises(RuntimeError, match="Could not determine GCP project ID from any source"):
            get_runtime_gcp_project_id()

    @patch("matrix.utils.kubernetes.get_gcp_project_from_config")
    def test_get_runtime_gcp_project_id_auto_detect(self, mock_get_project):
        """Test runtime project ID auto-detection when env var is not set."""
        mock_get_project.return_value = "auto-detected-project"

        with patch.dict(os.environ, {}, clear=True):
            result = get_runtime_gcp_project_id()

        assert result == "auto-detected-project"
        mock_get_project.assert_called_once()

    def test_get_runtime_mlflow_url_from_env(self):
        """Test MLflow URL retrieval from environment variable."""
        with patch.dict(os.environ, {"MLFLOW_URL": "https://custom.mlflow.com"}):
            result = get_runtime_mlflow_url()

        assert result == "https://custom.mlflow.com"

    def test_get_runtime_mlflow_url_prod_auto_detect(self):
        """Test MLflow URL auto-detection for production environment."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_runtime_mlflow_url("mtrx-hub-prod-sms")

        assert result == "https://mlflow.platform.prod.everycure.org/"

    def test_get_runtime_mlflow_url_dev_auto_detect(self):
        """Test MLflow URL auto-detection for development environment."""
        result = get_runtime_mlflow_url("mtrx-hub-dev-3of")

        assert result == "https://mlflow.platform.dev.everycure.org/"

    def test_get_runtime_gcp_bucket_from_env(self):
        """Test GCP bucket retrieval from environment variable."""
        with patch.dict(os.environ, {"RUNTIME_GCP_BUCKET": "custom-bucket"}):
            result = get_runtime_gcp_bucket()

        assert result == "custom-bucket"

    def test_get_runtime_gcp_bucket_prod_auto_detect(self):
        """Test GCP bucket auto-detection for production environment."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_runtime_gcp_bucket("mtrx-hub-prod-sms")

        assert result == "mtrx-us-central1-hub-prod-storage"

    def test_get_runtime_gcp_bucket_dev_auto_detect(self):
        """Test GCP bucket auto-detection for development environment."""
        result = get_runtime_gcp_bucket("mtrx-hub-dev-3of")

        assert result == "mtrx-us-central1-hub-dev-storage"

    @patch("matrix.utils.kubernetes.get_gcp_project_from_config")
    @patch("subprocess.check_output")
    @patch("subprocess.run")
    def test_can_talk_to_kubernetes_auto_detect_project(self, mock_run, mock_check_output, mock_get_project):
        """Test can_talk_to_kubernetes with auto-detected project."""
        mock_get_project.return_value = "mtrx-hub-dev-3of"
        mock_check_output.side_effect = [
            None,  # gcloud container clusters get-credentials
            "gke_mtrx-hub-dev-3of_us-central1_compute-cluster",  # kubectl config current-context
        ]
        mock_run.return_value = MagicMock(returncode=0)

        # Mock environment to avoid GitHub Actions check
        with patch.dict(os.environ, {}, clear=True):
            result = can_talk_to_kubernetes()

        assert result is True
        mock_get_project.assert_called_once()

    @patch("matrix.utils.kubernetes.get_gcp_project_from_config")
    @patch("subprocess.check_output")
    @patch("subprocess.run")
    def test_can_talk_to_kubernetes_explicit_project(self, mock_run, mock_check_output, mock_get_project):
        """Test can_talk_to_kubernetes with auto-detected project."""
        mock_get_project.return_value = "test-project"
        mock_check_output.side_effect = [
            None,  # gcloud container clusters get-credentials
            "gke_test-project_us-central1_compute-cluster",  # kubectl config current-context
        ]
        mock_run.return_value = MagicMock(returncode=0)

        # Mock environment to avoid GitHub Actions check
        with patch.dict(os.environ, {}, clear=True):
            # Test with auto-detected project
            result = can_talk_to_kubernetes()

        assert result is True
        mock_get_project.assert_called_once()


class TestBackwardCompatibility:
    """Test that the changes maintain backward compatibility."""

    @patch("matrix.utils.kubernetes.get_gcp_project_from_config")
    @patch("subprocess.check_output")
    @patch("subprocess.run")
    def test_existing_function_signatures_still_work(self, mock_run, mock_check_output, mock_get_project):
        """Test that the new function signature works correctly."""
        mock_get_project.return_value = "test-project"
        mock_check_output.side_effect = [
            None,  # gcloud container clusters get-credentials
            "gke_test-project_us-central1_compute-cluster",  # kubectl config current-context
        ]
        mock_run.return_value = MagicMock(returncode=0)

        # Mock environment to avoid GitHub Actions check
        with patch.dict(os.environ, {}, clear=True):
            # Test with new signature (auto-detection)
            result = can_talk_to_kubernetes()

        assert result is True
        mock_get_project.assert_called_once()

    @patch("requests.get")
    def test_get_gcp_project_from_metadata_success(self, mock_get):
        """Test successful project ID retrieval from GCP metadata server."""
        mock_response = Mock()
        mock_response.text = "test-project-123"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_gcp_project_from_metadata()

        assert result == "test-project-123"
        mock_get.assert_called_once_with(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
            timeout=5,
        )

    @patch("requests.get")
    def test_get_gcp_project_from_metadata_failure(self, mock_get):
        """Test metadata server failure handling."""
        mock_get.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to get project ID from GCP metadata server"):
            get_gcp_project_from_metadata()
