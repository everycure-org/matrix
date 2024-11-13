from matrix import settings
from kedro.pipeline import Pipeline, node, pipeline
import pandas as pd


def write_matrix_part(excel_df: pd.DataFrame, cols: list):
    return excel_df["matrix"].loc[cols]


def _create_sharing_pipeline(model: str) -> Pipeline:
    """Pipeline for sharing matrix results for a specific model with a medical team"""
    main_nodes = pipeline(
        [
            node(
                func=write_matrix_part,
                inputs=["matrix_generation.{model}.reporting.matrix_report", "params:sharing.matrix.stats_cols"],
                outputs="sharing.{model}.reporting.matrix_report_sheets_stats",
                name="write_matrix_stats_to_gsheets",
            ),
            node(
                func=write_matrix_part,
                inputs=["matrix_generation.{model}.reporting.matrix_report", "params:sharing.matrix.filter_cols"],
                outputs="sharing.{model}.reporting.matrix_report_sheets_filters",
                name="write_matrix_filter_to_gsheets",
            ),
            node(
                func=write_matrix_part,
                inputs=["matrix_generation.{model}.reporting.matrix_report", "params:sharing.matrix.pair_cols"],
                outputs="sharing.{model}.reporting.matrix_report_sheets_pairs",
                name="write_matrix_pairs_to_gsheets",
            ),
        ]
    )
    extra_nodes = pipeline(
        [
            node(
                func=lambda x: x,
                inputs="matrix_generation.{model}.reporting.matrix_report",
                outputs="sharing.{model}.reporting.matrix_report_gcp",
                name="copy_to_medical",
            ),
            node(
                func=lambda x: x,
                inputs="matrix_generation.{model}.reporting.matrix_report",
                outputs="sharing.{model}.reporting.matrix_metadata_sheets",
                name="write_metadata_to_gsheets",
            ),
            node(
                func=lambda x: x,
                inputs="matrix_generation.{model}.reporting.matrix_report",
                outputs="sharing.{model}.reporting.matrix_stats_sheets",
                name="write_stats_to_gsheets",
            ),
            node(
                func=lambda x: x,
                inputs="matrix_generation.{model}.reporting.matrix_report",
                outputs="sharing.{model}.reporting.matrix_legend_sheets",
                name="write_legend_to_gsheets",
            ),
        ]
    )
    return sum([*main_nodes, *extra_nodes])


def create_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for sharing results with medical team."""
    pipe = []
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    models_to_write = [model["model_name"] for model in models if model["share_result"]]
    for model in models_to_write:
        pipe.append(pipeline(_create_sharing_pipeline(model), tags=model))
    return sum([*pipe])
