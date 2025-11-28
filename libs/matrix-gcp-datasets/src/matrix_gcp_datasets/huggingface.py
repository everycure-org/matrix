from __future__ import annotations

import logging
import os
from typing import Any, Literal, Optional

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
                log.warn(
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
        hf_ds.push_to_hub(**push_kwargs)

    def _exists(self) -> bool:
        # Existence check is non-trivial against the Hub; skip and always write
        return False

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
