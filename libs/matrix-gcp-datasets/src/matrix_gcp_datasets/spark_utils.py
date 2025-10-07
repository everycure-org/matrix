# NOTE: This spark_utils.py was partially generated using AI assistance.
import logging
import os

import pyspark.sql as ps
from kedro.framework.context import KedroContext
from pyspark import SparkConf

logger = logging.getLogger(__name__)


def detect_gpus() -> int:
    """Detect the number of available GPUs.

    Returns:
        Number of GPUs available (0 if none detected).
    """
    try:
        import cupy as cp

        num_gpus = cp.cuda.runtime.getDeviceCount()
        logger.info(f"Detected {num_gpus} GPU(s) via CuPy")
        return num_gpus
    except (ImportError, Exception) as e:
        logger.debug(f"GPU detection via CuPy failed: {e}")

    # Fallback: Check CUDA_VISIBLE_DEVICES environment variable
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        try:
            num_gpus = len([d for d in cuda_visible.split(",") if d.strip()])
            logger.info(f"Detected {num_gpus} GPU(s) via CUDA_VISIBLE_DEVICES")
            return num_gpus
        except Exception as e:
            logger.debug(f"Failed to parse CUDA_VISIBLE_DEVICES: {e}")

    # Check if nvidia-smi is available
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            num_gpus = len(result.stdout.strip().split("\n"))
            logger.info(f"Detected {num_gpus} GPU(s) via nvidia-smi")
            return num_gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"GPU detection via nvidia-smi failed: {e}")

    logger.info("No GPUs detected")
    return 0


class SparkManager:
    """Simplified Spark session manager to avoid circular dependencies."""

    _spark_session: ps.SparkSession | None = None
    _kedro_context: KedroContext | None = None

    @classmethod
    def set_context(cls, context: KedroContext) -> None:
        """Set the Kedro context for configuration access."""
        cls._kedro_context = context

    @classmethod
    def initialize_spark(cls) -> None:
        """Initialize SparkSession if not already initialized."""
        if cls._kedro_context is None:
            # If no context is set, just return the active session or create a basic one
            logger.warning("No Kedro context set, using basic Spark session")
            if cls._spark_session is None:
                cls._spark_session = ps.SparkSession.builder.getOrCreate()
            return

        # Full initialization with Kedro context
        if cls._spark_session is None:
            # Clear any existing default session
            sess = ps.SparkSession.getActiveSession()
            if sess is not None:
                if "PYTEST_CURRENT_TEST" in os.environ:
                    cls._spark_session = sess
                    return
                logger.warning("Stopping existing Spark session to create a fresh one")
                sess.stop()

            # Load Spark configuration from Kedro
            parameters = cls._kedro_context.config_loader["spark"]

            # Production environment adjustments
            if "ARGO_NODE_ID" in os.environ:
                logger.warning("Production environment detected, removing service account auth configs")
                parameters = {
                    k: v for k, v in parameters.items() if not k.startswith("spark.hadoop.google.cloud.auth.service")
                }
            else:
                logger.info(f"Executing for environment: {cls._kedro_context.env}")
                logger.info(f"With ARGO_POD_UID set to: {os.environ.get('ARGO_NODE_ID', '')}")

            # Service account impersonation
            if os.environ.get("SPARK_IMPERSONATION_SERVICE_ACCOUNT") is not None:
                service_account = os.environ["SPARK_IMPERSONATION_SERVICE_ACCOUNT"]
                parameters["spark.hadoop.fs.gs.auth.impersonation.service.account"] = service_account
                logger.info(f"Using service account: {service_account} for spark impersonation")

            # Ensure all temporary operations use the mounted volume when SPARK_LOCAL_DIRS is set
            if os.environ.get("SPARK_LOCAL_DIRS") is not None:
                spark_local_dirs = os.environ["SPARK_LOCAL_DIRS"]
                logger.info(f"SPARK_LOCAL_DIRS detected: {spark_local_dirs}. Configuring additional temp paths.")

                # Override any relative temp paths to use the mounted volume
                temp_configs = {
                    "spark.sql.warehouse.dir": f"{spark_local_dirs}/spark-warehouse",
                    "spark.sql.streaming.checkpointLocation": f"{spark_local_dirs}/checkpoints",
                    "java.io.tmpdir": spark_local_dirs,
                    "spark.driver.host.tmpdir": spark_local_dirs,
                    "spark.executor.host.tmpdir": spark_local_dirs,
                }

                for key, value in temp_configs.items():
                    parameters[key] = value
                    logger.info(f"Setting {key} to {value}")

            # Configure GPU resources if available
            num_gpus = detect_gpus()
            if num_gpus > 0:
                logger.info(f"Configuring Spark for {num_gpus} GPU(s)")
                gpu_configs = {
                    "spark.executor.resource.gpu.amount": num_gpus,
                    "spark.task.resource.gpu.amount": num_gpus,
                    "spark.rapids.sql.enabled": "true",
                    "spark.rapids.memory.pinnedPool.size": "2G",
                    "spark.python.worker.reuse": "true",
                    # Enable GPU-accelerated operations
                    "spark.sql.execution.arrow.pyspark.enabled": "true",
                }

                # Only override if not already set in config
                for key, value in gpu_configs.items():
                    if key not in parameters:
                        parameters[key] = value
                        logger.info(f"Setting GPU config {key} to {value}")
            else:
                logger.info("No GPUs detected, using CPU-only Spark configuration")

            logger.info(f"Starting Spark session with parameters: {parameters}")
            spark_conf = SparkConf().setAll(parameters.items())

            # Create and set configured session as default
            cls._spark_session = (
                ps.SparkSession.builder.appName(cls._kedro_context.project_path.name)
                .config(conf=spark_conf)
                .getOrCreate()
            )
        else:
            logger.debug("SparkSession already initialized")

    @classmethod
    def get_or_create_session(cls) -> ps.SparkSession:
        """Get existing session or create a new one."""
        if cls._spark_session is None:
            cls.initialize_spark()
        return cls._spark_session
