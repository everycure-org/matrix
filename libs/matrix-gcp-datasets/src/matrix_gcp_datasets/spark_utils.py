# NOTE: This spark_utils.py was partially generated using AI assistance.
import logging
import os

import pyspark.sql as ps
from kedro.framework.context import KedroContext
from pyspark import SparkConf

logger = logging.getLogger(__name__)


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
