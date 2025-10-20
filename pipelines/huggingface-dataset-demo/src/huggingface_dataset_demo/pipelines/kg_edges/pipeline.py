from __future__ import annotations

from kedro.pipeline import Pipeline, node  # type: ignore[import-not-found]

from .nodes import generate_data_and_save_to_hf, read_hf_datasets_and_write_jsonl


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            # Node 1: Generate data and return 3 datasets (Spark, Pandas, Polars)
            # All 3 get saved to Hugging Face Hub automatically by Kedro
            node(
                func=generate_data_and_save_to_hf,
                inputs=None,
                outputs=["kg_edges_spark_hf", "kg_edges_pandas_hf", "kg_edges_polars_hf"],
                name="generate_data_and_save_to_hf_node",
            ),
            # Node 2: Read 3 datasets from Hugging Face Hub and write as JSONL
            node(
                func=read_hf_datasets_and_write_jsonl,
                inputs=["kg_edges_spark_hf_read", "kg_edges_pandas_hf_read", "kg_edges_polars_hf_read"],
                outputs=["kg_edges_spark_jsonl", "kg_edges_pandas_jsonl", "kg_edges_polars_jsonl"],
                name="read_hf_datasets_and_write_jsonl_node",
            ),
        ]
    )
