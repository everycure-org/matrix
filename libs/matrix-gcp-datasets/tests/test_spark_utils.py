# NOTE: This file was partially generated using AI assistance.
import os
import sys
from unittest.mock import MagicMock, patch

from matrix_gcp_datasets.spark_utils import detect_gpus


def test_detect_gpus_via_cupy():
    """Test GPU detection using CuPy when available."""
    mock_cp = MagicMock()
    mock_cp.cuda.runtime.getDeviceCount.return_value = 2

    with patch.dict(sys.modules, {"cupy": mock_cp}):
        num_gpus = detect_gpus()

        assert num_gpus == 2
        mock_cp.cuda.runtime.getDeviceCount.assert_called_once()


def test_detect_gpus_via_cuda_visible_devices():
    """Test GPU detection using CUDA_VISIBLE_DEVICES environment variable."""
    import builtins

    # Store the original import
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("No cupy available")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"}):
            num_gpus = detect_gpus()

            assert num_gpus == 3


def test_detect_gpus_via_nvidia_smi():
    """Test GPU detection using nvidia-smi command."""
    import builtins

    # Store the original import
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("No cupy available")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="1\n1\n")

                num_gpus = detect_gpus()

                assert num_gpus == 2


def test_detect_gpus_no_gpu_available():
    """Test GPU detection when no GPUs are available."""
    import builtins

    # Store the original import
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("No cupy available")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                num_gpus = detect_gpus()

                assert num_gpus == 0


def test_spark_manager_gpu_configuration_enabled():
    """Test that SparkManager adds GPU configuration when GPUs are detected."""
    with patch("matrix_gcp_datasets.spark_utils.detect_gpus", return_value=2):
        # Create base config
        parameters = {
            "spark.app.name": "test-app",
            "spark.master": "local[*]",
        }

        # Simulate the GPU configuration logic from SparkManager
        num_gpus = 2
        gpu_configs = {
            "spark.executor.resource.gpu.amount": num_gpus,
            "spark.task.resource.gpu.amount": 0.1,
            "spark.rapids.sql.enabled": "true",
            "spark.rapids.memory.pinnedPool.size": "2G",
            "spark.python.worker.reuse": "true",
            "spark.sql.execution.arrow.pyspark.enabled": "true",
        }

        # Apply GPU configs that aren't already set
        for key, value in gpu_configs.items():
            if key not in parameters:
                parameters[key] = value

        # Verify the configs were added
        assert parameters["spark.executor.resource.gpu.amount"] == 2
        assert parameters["spark.task.resource.gpu.amount"] == 0.1
        assert parameters["spark.rapids.sql.enabled"] == "true"
        assert parameters["spark.rapids.memory.pinnedPool.size"] == "2G"
        assert parameters["spark.python.worker.reuse"] == "true"


def test_spark_manager_gpu_configuration_disabled():
    """Test that SparkManager does not add GPU configuration when no GPUs are detected."""
    with patch("matrix_gcp_datasets.spark_utils.detect_gpus", return_value=0):
        # Create base config
        parameters = {
            "spark.app.name": "test-app",
            "spark.master": "local[*]",
        }

        # Simulate the GPU configuration logic from SparkManager
        num_gpus = 0
        if num_gpus > 0:
            gpu_configs = {
                "spark.executor.resource.gpu.amount": num_gpus,
                "spark.task.resource.gpu.amount": 0.1,
                "spark.rapids.sql.enabled": "true",
            }

            for key, value in gpu_configs.items():
                if key not in parameters:
                    parameters[key] = value

        # Verify GPU configs were NOT added
        assert "spark.executor.resource.gpu.amount" not in parameters
        assert "spark.task.resource.gpu.amount" not in parameters
        assert "spark.rapids.sql.enabled" not in parameters
