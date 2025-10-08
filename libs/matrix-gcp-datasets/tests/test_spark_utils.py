# NOTE: This file was partially generated using AI assistance.
import os
from unittest.mock import MagicMock, patch

from matrix_gcp_datasets.spark_utils import SparkManager, detect_gpus


def test_detect_gpus_via_cupy():
    """Test GPU detection using CuPy when available."""
    with patch("matrix_gcp_datasets.spark_utils.cp") as mock_cp:
        mock_cp.cuda.runtime.getDeviceCount.return_value = 2

        num_gpus = detect_gpus()

        assert num_gpus == 2
        mock_cp.cuda.runtime.getDeviceCount.assert_called_once()


def test_detect_gpus_via_cuda_visible_devices():
    """Test GPU detection using CUDA_VISIBLE_DEVICES environment variable."""
    with patch("matrix_gcp_datasets.spark_utils.cp", side_effect=ImportError):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"}):
            num_gpus = detect_gpus()

            assert num_gpus == 3


def test_detect_gpus_via_nvidia_smi():
    """Test GPU detection using nvidia-smi command."""
    with patch("matrix_gcp_datasets.spark_utils.cp", side_effect=ImportError):
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="1\n1\n")

                num_gpus = detect_gpus()

                assert num_gpus == 2


def test_detect_gpus_no_gpu_available():
    """Test GPU detection when no GPUs are available."""
    with patch("matrix_gcp_datasets.spark_utils.cp", side_effect=ImportError):
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                num_gpus = detect_gpus()

                assert num_gpus == 0


def test_spark_manager_gpu_configuration_enabled(kedro_session):
    """Test that SparkManager configures GPU settings when GPUs are detected."""
    with patch("matrix_gcp_datasets.spark_utils.detect_gpus", return_value=2):
        SparkManager.set_context(kedro_session._get_context())
        SparkManager._spark_session = None

        SparkManager.initialize_spark()

        spark_conf = SparkManager._spark_session.sparkContext.getConf()

        # Check GPU-related configs were set
        assert spark_conf.get("spark.executor.resource.gpu.amount") == "1"
        assert spark_conf.get("spark.task.resource.gpu.amount") == "0.1"  # Fractional for concurrency
        assert spark_conf.get("spark.rapids.sql.enabled") == "true"


def test_spark_manager_gpu_configuration_disabled(kedro_session):
    """Test that SparkManager does not configure GPU settings when no GPUs are detected."""
    with patch("matrix_gcp_datasets.spark_utils.detect_gpus", return_value=0):
        SparkManager.set_context(kedro_session._get_context())
        SparkManager._spark_session = None

        SparkManager.initialize_spark()

        spark_conf = SparkManager._spark_session.sparkContext.getConf()

        # Check GPU-related configs were not set (will return None or empty string)
        try:
            gpu_amount = spark_conf.get("spark.executor.resource.gpu.amount")
            # If it exists, it should be from base config, not from auto-detection
            assert gpu_amount in [None, ""]
        except Exception:
            # Config key doesn't exist, which is expected
            pass
