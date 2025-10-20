from __future__ import annotations

from kedro.pipeline import Pipeline, node  # type: ignore[import-not-found]

from .nodes import build_kg_edges_spark, spark_to_jsonl


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_kg_edges_spark,
                inputs=None,
                outputs="kg_edges_spark",
                name="build_kg_edges_spark_node",
            ),
            node(
                func=spark_to_jsonl,
                inputs="kg_edges_spark",
                outputs="kg_edges_local_jsonl",
                name="read_hf_and_write_jsonl",
            ),
        ]
    )
