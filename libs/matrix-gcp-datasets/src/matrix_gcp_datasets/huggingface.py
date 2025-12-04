from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal, Optional

import requests
from datasets import load_dataset
from kedro.io import AbstractDataset
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)


class HFIterableDatasetConfig(BaseModel):
    """Configuration for HFIterableDataset."""

    repo_id: str
    split: str = Field(default="train")
    config_name: Optional[str] = None
    private: bool = Field(default=False)
    token_key: str = Field(default="HF_TOKEN")
    token: Optional[str] = None
    dataframe_type: Literal["spark", "polars", "pandas"] = Field(default="spark")
    data_dir: Optional[str] = None
    

    def build_push_kwargs(self, token: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "repo_id": self.repo_id,
            "token": token,
            "split": self.split,
        }
        if self.config_name:
            kwargs["config_name"] = self.config_name
        if self.private:
            kwargs["private"] = True
        if self.data_dir:
            kwargs["data_dir"] = self.data_dir  # type: ignore[dict-item]
        return kwargs

    def __repr__(self) -> str:
        tok = self.token
        return (
            f"HFIterableDatasetConfig(repo_id={self.repo_id!r}, split={self.split!r}, "
            f"config_name={self.config_name!r}, private={self.private!r}, "
            f"token_key={self.token_key!r}, token={'***' if tok else None}, "
            f"dataframe_type={self.dataframe_type!r}, data_dir={self.data_dir!r})"
        )


class HFIterableDataset(AbstractDataset):
    """
    Kedro dataset that accepts a Spark DataFrame on save() and pushes it to
    a Hugging Face Hub dataset using datasets.Dataset.from_spark.

    This dataset is write-only. Loading is not implemented.

    Required dataset config parameters:
      - repo_id: str, e.g. "everycure/test-matrix-kg"

    Optional parameters:
      - split: str, split name to push (default: "train")
      - config_name: str | None, HF config (subset) name
      - private: bool, whether to push to a private repo (default: False)
      - credentials: str, Kedro credentials key to resolve token from
      - token_key: str, key inside the credentials mapping (default: "HF_TOKEN")
      - token: str, direct token override (discouraged; prefer credentials)
    """

    def __init__(
        self,
        repo_id: str,
        data_dir: Optional[str] = None,
        split: str = "train",
        config_name: Optional[str] = None,
        private: bool = False,
        credentials: Optional[dict[str, Any]] = None,
        token_key: str = "HF_TOKEN",
        token: Optional[str] = None,
        dataframe_type: Literal["spark", "polars", "pandas"] = "spark",
    ) -> None:
        super().__init__()
        try:
            self.config = HFIterableDatasetConfig(
                repo_id=repo_id,
                data_dir=data_dir,
                split=split,
                config_name=config_name,
                private=private,
                token_key=token_key,
                token=token,
                dataframe_type=dataframe_type,
            )
        except ValidationError as exc:
            raise ValueError(f"Invalid HFIterableDataset configuration: {exc}")

        # Kedro injects resolved credentials dict here when `credentials:` is set in catalog
        self._credentials: Optional[dict[str, Any]] = credentials

    def _describe(self) -> dict[str, Any]:
        return {"config": repr(self.config)}

    def _load(self) -> Any:
        from datasets import load_dataset

        token = self._resolve_token()
        ds = load_dataset(self.config.repo_id, split=self.config.split, token=token)
        df_type = self.config.dataframe_type
        if df_type == "spark":
            try:
                from pyspark.sql import SparkSession
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Spark is not installed but dataframe_type='spark'.") from exc
            spark = SparkSession.builder.getOrCreate()
            # Prefer Arrow table -> Spark (Spark 4+), fallback to pandas
            try:
                arrow_tbl = ds.to_table()  # pyarrow.Table
                return spark.createDataFrame(arrow_tbl)  # Spark 4.0+
            except Exception:
                # Enable Arrow for pandas conversions
                try:
                    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
                except Exception:
                    pass
                log.warning(
                    f"Converting to pandas from Hugging Face dataset as fallback. Upgrade to Spark 4+ to avoid this."
                )
                pdf = ds.to_pandas()
                return spark.createDataFrame(pdf)
        if df_type == "polars":
            try:
                return ds.to_polars()
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Polars not available for conversion. `uv add polars`.") from exc
        if df_type == "pandas":
            return ds.to_pandas()
        raise ValueError(f"Unsupported dataframe_type: {df_type}")

    def _save(self, data: Any) -> None:
        # Local import to avoid importing heavy deps unless used
        from datasets import Dataset

        token = self._resolve_token()

        # Route based on configured dataframe type
        df_type = self.config.dataframe_type
        if df_type == "spark":
            # Expecting a Spark DataFrame
            hf_ds = Dataset.from_spark(data)
        elif df_type == "polars":
            # Use Dataset.from_polars for direct Polars support
            try:
                import polars as pl
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Polars requested but not installed. Please `uv add polars`.") from exc
            if not isinstance(data, pl.DataFrame):
                raise TypeError("Configured dataframe_type='polars' but got different object")
            hf_ds = Dataset.from_polars(data)
        elif df_type == "pandas":
            # Avoid importing pandas types to keep linters happy without stubs
            try:
                hf_ds = Dataset.from_pandas(data)
            except Exception as exc:  # pragma: no cover
                raise TypeError(
                    "Configured dataframe_type='pandas' but the input is not a pandas DataFrame "
                    "or pandas is not installed. Please `uv add pandas` and pass a pandas.DataFrame."
                ) from exc
        else:  # pragma: no cover - guarded by pydantic literal
            raise ValueError(f"Unsupported dataframe_type: {df_type}")

        # Push to hub; tolerate older signatures lacking data_dir
        push_kwargs = self.config.build_push_kwargs(token or "")
        ci = hf_ds.push_to_hub(**push_kwargs)
        self._verify_hf_upload(self.config.repo_id, ci.oid)

    def _resolve_token(self) -> str | None:
        # Precedence: explicit token override > credentials > env var
        if self.config.token:
            return self.config.token

        # Kedro passes resolved credentials mapping
        if isinstance(self._credentials, dict):
            maybe = self._credentials.get(self.config.token_key)
            if isinstance(maybe, str) and maybe:
                return maybe

        env_token = os.getenv("HF_TOKEN")
        if env_token:
            return env_token

        raise ValueError(
            "Hugging Face token not found. Provide via catalog credentials, "
            "explicit dataset `token`, or HF_TOKEN environment variable."
        )

    def _get_latest_sha(self, dataset_id, revision="main"):
        """Fetch the latest commit SHA from the HuggingFace Hub."""
        url = f"https://huggingface.co/api/datasets/{dataset_id}/revision/{revision}"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()["sha"]


    def _list_hub_files(self, dataset_id, revision="main"):
        """List files in the dataset repository on HF Hub."""
        url = f"https://huggingface.co/api/datasets/{dataset_id}/tree/{revision}"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()


    def _wait_for_sha(self, dataset_id, expected_sha, timeout=300, interval=5):
        """Wait for a specified amount of timeuntil the Hub reports the expected SHA."""
        start = time.time()
        while True:
            current_sha = self._get_latest_sha(dataset_id)
            if current_sha == expected_sha:
                log.info(f"✓ SHA match confirmed: {current_sha}")
                return True

            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Timed out waiting for SHA to update. "
                    f"Current SHA: {current_sha}, Expected: {expected_sha}"
                )

            log.info(f"Waiting for SHA update... (current={current_sha}, expected={expected_sha})")
            time.sleep(interval)


    def _verify_stream_load(self, dataset_id, revision="main"):
        """Try streaming one row from the dataset to confirm it is readable."""
        try:
            ds = load_dataset(dataset_id, split="train", streaming=True, revision=revision)
            first_row = next(iter(ds))
            log.info("✓ Streaming load succeeded. Sample row:", first_row)
            return True
        except Exception as e:
            raise RuntimeError(f"Streaming load failed: {e}")


    def _verify_hf_upload(self, dataset_id, pushed_sha):
        """Verify if the upload to Hugging Face Hub was successful."""
        log.info("\n=== Step 1: Checking commit SHA ===")
        self._wait_for_sha(dataset_id, pushed_sha)

        log.info("\n=== Step 2: Checking file exist on HF Hub ===")
        files = self._list_hub_files(dataset_id)
        if not files:
            raise RuntimeError("No files found on Hub — upload may have failed!")
        log.info(f"✓ Found {len(files)} files on Hub.")
        for f in files:
            log.info(f" - {f['path']} ({f['size']} bytes)")

        log.info("\n=== Step 3 (optional): Checking streaming availability ===")
        self._verify_stream_load(dataset_id)

        log.info(f"\n🎉 All checks passed. Upload of {dataset_id} to Hugging Face Hub is verified.")

