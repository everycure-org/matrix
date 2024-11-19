"""This module creates a pipeline for the indicted/nonindicted pairs analysis."""

from kedro.pipeline import Pipeline, node, pipeline
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.normalize_curies,
                inputs=["input_df", "norm_cache", "params:normalization"],
                outputs=["updated_norm_cache", "failed_ids"],
                name="normalize_curies_node",
            ),
            node(
                func=nodes.apply_normalization_to_df,
                inputs=["input_df", "updated_norm_cache"],
                outputs="normalized_df",
                name="apply_normalization_node",
            ),
            node(
                func=nodes.extract_officially_indicated_pairs,
                inputs=["normalized_df", "tsv_df", "approval_cache", "params:extraction"],
                outputs=["updated_approval_cache", "officially_indicated_pairs"],
                name="extract_officially_indicated_pairs_node",
            ),
            node(
                func=nodes.annotate_indicated_pairs,
                inputs=["normalized_df", "officially_indicated_pairs"],
                outputs="indicated_df",
                name="annotate_indicated_pairs_node",
            ),
            node(
                func=nodes.filter_approved_rows,
                inputs="indicated_df",
                outputs="approved_df",
                name="filter_approved_rows_node",
            ),
            node(
                func=nodes.compute_statistics,
                inputs=["approved_df"],
                outputs="statistics",
                name="compute_statistics_node",
            ),
        ],
    )
