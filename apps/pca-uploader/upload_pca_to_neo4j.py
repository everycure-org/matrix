#!/usr/bin/env python3
"""
Clean, robust PCA coordinates uploader for Neo4j with checkpoint support.

Staff Engineer Implementation:
- Kedro-first design with catalog integration
- Checkpoint/resume functionality for interrupted uploads
- Clean separation of concerns
- Modern CLI with Click
- Efficient batching with progress tracking
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import click
import pandas as pd
from dotenv import load_dotenv
from joblib import Memory
from neo4j import GraphDatabase, Session
from tqdm import tqdm

# NOTE: This script was partially generated using AI assistance.

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure joblib memory cache
memory = Memory(".cache", verbose=0)


@dataclass
class UploadConfig:
    """Configuration for PCA upload process."""

    neo4j_host: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    batch_size: int = 2000
    max_workers: int = 8
    checkpoint_file: str = ".pca_upload_checkpoint.json"
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_timeout: int = 30


@dataclass
class Checkpoint:
    """Checkpoint data for resuming interrupted uploads."""

    total_records: int
    completed_batches: int
    successful_updates: int
    failed_batches: int

    def save(self, filepath: str) -> None:
        """Save checkpoint to file."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, filepath: str) -> Optional["Checkpoint"]:
        """Load checkpoint from file."""
        if not Path(filepath).exists():
            return None
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    @classmethod
    def delete(cls, filepath: str) -> None:
        """Delete checkpoint file."""
        Path(filepath).unlink(missing_ok=True)


def load_pca_data(catalog, dataset_name: str) -> pd.DataFrame:
    """Load and prepare PCA data from Kedro catalog using efficient Spark operations."""
    logger.info(f"Loading PCA data from catalog: {dataset_name}")

    # Load from catalog
    df = catalog.load(dataset_name)

    # Extract PCA coordinates using Spark operations
    from pyspark.sql.functions import col, element_at

    # Extract pca_0 and pca_1 from the pca_embedding array
    df_with_coords = df.select(
        "id",
        element_at("pca_embedding", 1).alias("pca_0"),
        element_at("pca_embedding", 2).alias("pca_1"),
    )

    # Filter out rows where PCA coordinates are null
    df_filtered = df_with_coords.filter(
        col("pca_0").isNotNull() & col("pca_1").isNotNull()
    )

    # Convert to Pandas only for the final result (small dataset)
    result_df = df_filtered.toPandas()
    logger.info(
        f"Extracted PCA coordinates for {len(result_df)} nodes using Spark operations"
    )
    return result_df


# Create cached version for expensive operations
# Ignore the unpickleable `catalog` when computing the cache key
load_pca_data_cached = memory.cache(load_pca_data, ignore=["catalog"])


class PCAPipeline:
    """Clean PCA upload pipeline with checkpoint support."""

    def __init__(self, config: UploadConfig):
        self.config = config
        self.driver = None

    def __enter__(self):
        """Context manager entry."""
        load_dotenv()
        self.driver = GraphDatabase.driver(
            self.config.neo4j_host,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
            connection_timeout=self.config.connection_timeout,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.driver:
            self.driver.close()

    def create_batches(self, df: pd.DataFrame, start_batch: int = 0) -> list:
        """Create batches for upload, optionally starting from a checkpoint."""
        total_batches = (len(df) + self.config.batch_size - 1) // self.config.batch_size

        batches = []
        for i in range(start_batch, total_batches):
            start_idx = i * self.config.batch_size
            end_idx = min((i + 1) * self.config.batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            batch_data = [
                {
                    "id": row["id"],
                    "pca_0": float(row["pca_0"]),
                    "pca_1": float(row["pca_1"]),
                }
                for _, row in batch_df.iterrows()
            ]
            batches.append((i, batch_data))

        return batches

    def update_batch(self, session: Session, batch_data: list) -> int:
        """Update a batch of nodes with PCA coordinates."""
        query = """
        UNWIND $batch AS row
        MATCH (n:Entity {id: row.id})
        SET n.pca_0 = row.pca_0, n.pca_1 = row.pca_1
        RETURN count(n) as updated
        """

        try:
            result = session.run(query, batch=batch_data)
            return result.single()["updated"]
        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            # Log more details for debugging
            if "object cannot be re-sized" in str(e):
                logger.error(
                    "This error typically indicates a session reuse issue - check thread safety"
                )
            elif "Connection" in str(e):
                logger.error("Connection issue detected - may need to retry")
            return 0

    def upload_with_checkpoints(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Upload PCA data with checkpoint support."""
        # Load existing checkpoint
        checkpoint = Checkpoint.load(self.config.checkpoint_file)

        if checkpoint:
            logger.info(
                f"Resuming from checkpoint: {checkpoint.completed_batches} batches completed"
            )
            start_batch = checkpoint.completed_batches
            successful_updates = checkpoint.successful_updates
            failed_batches = checkpoint.failed_batches
        else:
            start_batch = 0
            successful_updates = 0
            failed_batches = 0

        # Create batches
        batches = self.create_batches(df, start_batch)
        total_batches = (len(df) + self.config.batch_size - 1) // self.config.batch_size

        if not batches:
            logger.info("All batches already completed!")
            return successful_updates, failed_batches

        logger.info(
            f"Processing {len(batches)} remaining batches (total: {total_batches})"
        )

        # Upload with progress tracking
        with tqdm(
            total=total_batches, initial=start_batch, desc="Uploading PCA coordinates"
        ) as pbar:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit batches - each thread will create its own session
                future_to_batch = {
                    executor.submit(
                        self._update_batch_with_session, batch_data
                    ): batch_idx
                    for batch_idx, batch_data in batches
                }

                # Process results
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]

                    try:
                        updated_count = future.result()
                        successful_updates += updated_count

                    except Exception as e:
                        logger.error(f"Batch {batch_idx} failed: {e}")
                        failed_batches += 1

                    # Update progress and checkpoint
                    completed_batches = batch_idx + 1
                    pbar.update(1)
                    pbar.set_postfix(
                        {"Updated": successful_updates, "Failed": failed_batches}
                    )

                    # Save checkpoint periodically
                    if completed_batches % 10 == 0:
                        checkpoint = Checkpoint(
                            total_records=len(df),
                            completed_batches=completed_batches,
                            successful_updates=successful_updates,
                            failed_batches=failed_batches,
                        )
                        checkpoint.save(self.config.checkpoint_file)

        # Final cleanup
        Checkpoint.delete(self.config.checkpoint_file)

        return successful_updates, failed_batches

    def _update_batch_with_session(self, batch_data: list) -> int:
        """Update a batch of nodes with PCA coordinates using a thread-local session."""
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay

        for attempt in range(max_retries):
            try:
                # Create a new session for this thread
                with self.driver.session(
                    database=self.config.neo4j_database
                ) as session:
                    return self.update_batch(session, batch_data)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Batch failed on attempt {attempt + 1}, retrying in {retry_delay}s: {e}"
                    )
                    import time

                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Batch failed after {max_retries} attempts: {e}")
                    raise

    def test_connection(self) -> bool:
        """Test Neo4j connection and return health status."""
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                result = session.run("RETURN 'world' as test").single()
                logger.info(f"✓ Neo4j connection successful - Hello: {result['test']}")
                return True
        except Exception as e:
            logger.error(f"✗ Neo4j connection failed: {e}")
            return False

    def run(self, catalog, dataset_name: str) -> None:
        """Main execution pipeline."""
        logger.info("Starting PCA upload pipeline")

        # Test connection
        if not self.test_connection():
            raise ConnectionError("Failed to connect to Neo4j database")

        # Load and process data
        df = load_pca_data_cached(catalog, dataset_name)
1
        # Upload with checkpoints
        successful_updates, failed_batches = self.upload_with_checkpoints(df)

        # Summary
        logger.info("Upload completed!")
        logger.info(f"Successfully updated: {successful_updates} nodes")
        logger.info(f"Failed batches: {failed_batches}")
        logger.info(f"Total records processed: {len(df)}")


def create_config() -> UploadConfig:
    """Create upload configuration from environment variables."""
    return UploadConfig(
        neo4j_host=os.getenv("NEO4J_HOST", "bolt://127.0.0.1:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "admin"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "analytics"),
    )


# API Functions for backward compatibility and programmatic use
def upload_pca_coordinates_from_catalog(
    catalog,
    dataset_name: str = "embeddings.reporting.topological_pca",
    batch_size: int = 2000,
    max_workers: int = 8,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    connection_timeout: int = 30,
) -> None:
    """Upload PCA coordinates using Kedro catalog (main API)."""
    config = create_config()
    config.batch_size = batch_size
    config.max_workers = max_workers
    config.max_retries = max_retries
    config.retry_delay = retry_delay
    config.connection_timeout = connection_timeout

    with PCAPipeline(config) as pipeline:
        pipeline.run(catalog, dataset_name)


# CLI Interface
@click.group()
def cli():
    """PCA Uploader - Upload PCA coordinates to Neo4j with checkpoint support."""
    pass


@cli.command()
def status():
    """Show current upload status and checkpoint information."""
    checkpoint = Checkpoint.load(".pca_upload_checkpoint.json")

    if checkpoint:
        logger.info("Checkpoint found:")
        logger.info(f"  Total records: {checkpoint.total_records}")
        logger.info(f"  Completed batches: {checkpoint.completed_batches}")
        logger.info(f"  Successful updates: {checkpoint.successful_updates}")
        logger.info(f"  Failed batches: {checkpoint.failed_batches}")

        progress = (
            (checkpoint.completed_batches * 2000) / checkpoint.total_records * 100
        )
        logger.info(f"  Progress: {progress:.1f}%")
    else:
        logger.info("No checkpoint found - no upload in progress")


@cli.command()
def clean():
    """Clean up checkpoint files."""
    Checkpoint.delete(".pca_upload_checkpoint.json")
    logger.info("Checkpoint cleaned")


@cli.command()
@click.option(
    "--dataset-name",
    "-d",
    default="embeddings.reporting.topological_pca",
    help="Kedro dataset name to load PCA data from",
)
@click.option("--batch-size", "-b", default=2000, help="Batch size for Neo4j updates")
@click.option(
    "--max-workers", "-w", default=8, help="Maximum number of parallel workers"
)
@click.option(
    "--resume/--no-resume", default=True, help="Resume from checkpoint if available"
)
@click.option(
    "--project-path",
    "-p",
    default="pipelines/matrix",
    help="Path to Kedro project directory (relative to current directory)",
)
@click.option(
    "--env",
    "-e",
    default="cloud",
    help="Kedro environment to use (default: cloud)",
)
@click.option(
    "--gcp-keyfile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a GCP service account JSON keyfile to use for GCS access (sets GOOGLE_APPLICATION_CREDENTIALS)",
)
@click.option(
    "--max-retries",
    default=3,
    help="Maximum number of retries for failed batches",
)
@click.option(
    "--retry-delay",
    default=1.0,
    help="Initial delay between retries in seconds",
)
@click.option(
    "--connection-timeout",
    default=30,
    help="Neo4j connection timeout in seconds",
)
def upload(
    dataset_name: str,
    batch_size: int,
    max_workers: int,
    resume: bool,
    project_path: str,
    env: str,
    gcp_keyfile: Optional[Path],
    max_retries: int,
    retry_delay: float,
    connection_timeout: int,
):
    """Upload PCA coordinates from Kedro catalog to Neo4j."""

    if not resume:
        Checkpoint.delete(".pca_upload_checkpoint.json")
        logger.info("Checkpoint cleared - starting fresh upload")

    try:
        # Import Kedro dynamically
        from kedro.framework.session import KedroSession
        from kedro.framework.startup import bootstrap_project

        # Resolve and validate project path
        project_path = Path(project_path).resolve()
        if not project_path.exists():
            raise FileNotFoundError(
                f"Kedro project not found at {project_path}. "
                "Please check the --project-path parameter."
            )

        # Load project environment variables before Kedro config resolution
        load_dotenv(project_path / ".env.defaults", override=False)
        load_dotenv(project_path / ".env", override=False)

        # Optionally set explicit ADC credential file for local runs
        if gcp_keyfile is not None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(gcp_keyfile)

        # Bootstrap to set PACKAGE_NAME and other settings
        bootstrap_project(project_path)

        # Create Kedro session using explicit project path and env
        # Ensure process CWD is the Kedro project so relative spark.jars resolve correctly
        original_cwd = os.getcwd()
        os.chdir(project_path)
        try:
            with KedroSession.create(
                project_path=str(project_path), env=env
            ) as session:
                catalog = session.load_context().catalog

                config = create_config()
                config.batch_size = batch_size
                config.max_workers = max_workers
                config.max_retries = max_retries
                config.retry_delay = retry_delay
                config.connection_timeout = connection_timeout

                with PCAPipeline(config) as pipeline:
                    pipeline.run(catalog, dataset_name)
        finally:
            os.chdir(original_cwd)

    except ImportError:
        logger.error(
            "Kedro not available. Please install Kedro or use the API functions."
        )
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


if __name__ == "__main__":
    cli()
