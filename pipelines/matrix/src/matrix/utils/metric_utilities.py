"""
Utilities for logging metrics about datasets during pipeline execution.

This module provides decorators and functions to automatically log metrics
about datasets as they flow through a Kedro pipeline.
"""
# NOTE: This file was partially generated using AI assistance.

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Union

import mlflow
import pandas as pd
import pyspark.sql as ps

logger = logging.getLogger(__name__)

# Type aliases
Dataset = Union[pd.DataFrame, ps.DataFrame]
MetricFunction = Callable[[Dataset], None]


def log_metric(namespace: str, value: Union[int, float], context: str = None) -> None:
    """Log a metric to MLflow with proper context.

    Args:
        context: Context (usually node name) for the metric
        name: Name of the metric
        value: Value to log
    """
    context_str = f" ({context})" if context else ""
    key_string = f"{namespace}/{value}{context_str}"
    logger.info(f"{key_string}: {value}")
    mlflow.log_metric(key_string, value)


def is_spark_dataframe(df: Dataset) -> bool:
    """Check if dataset is a Spark DataFrame."""
    return isinstance(df, ps.DataFrame)


def is_pandas_dataframe(df: Dataset) -> bool:
    """Check if dataset is a Pandas DataFrame."""
    return isinstance(df, pd.DataFrame)


def log_count(name: str, df: Dataset) -> None:
    """Log the number of rows in a dataset.

    Args:
        context: The context (usually node name) for the metric
        name: Name of the dataset being logged
        df: Dataset to analyze (pandas or spark DataFrame)
    """
    count = df.count() if is_spark_dataframe(df) else len(df)
    log_metric(name, count)


def log_distribution(name: str, df: Dataset) -> None:
    """Log basic statistics for numerical columns.

    Args:
        context: The context (usually node name) for the metric
        name: Name of the dataset being logged
        df: Dataset to analyze (pandas or spark DataFrame)
    """
    if is_spark_dataframe(df):
        summary = df.summary("count", "mean", "stddev", "min", "max").collect()
        # Convert spark summary to dict for logging
        for col in df.columns:
            for stat in ["mean", "stddev", "min", "max"]:
                value = float(next(row[col] for row in summary if row["summary"] == stat))
                log_metric(name, f"{col}_{stat}", value)
    elif is_pandas_dataframe(df):
        # For pandas, use describe()
        desc = df.describe()
        for col in df.select_dtypes(include=["number"]).columns:
            for stat in ["mean", "std", "min", "max"]:
                log_metric(name, f"{col}_{stat}", desc[col][stat])
    else:
        raise ValueError("Unsupported dataset type. Expected pandas or spark DataFrame.")


def log_null_count(name: str, df: Dataset) -> None:
    """Log count of null values in each column.

    Args:
        context: The context (usually node name) for the metric
        name: Name of the dataset being logged
        df: Dataset to analyze (pandas or spark DataFrame)
    """
    if is_spark_dataframe(df):
        for col in df.columns:
            null_count = df.filter(df[col].isNull()).count()
            log_metric(name, f"{col}_nulls", null_count)
    elif is_pandas_dataframe(df):
        for col in df.columns:
            null_count = df[col].isnull().sum()
            log_metric(name, f"{col}_nulls", null_count)
    else:
        raise ValueError("Unsupported dataset type. Expected pandas or spark DataFrame.")


def log_metrics(
    inputs: Dict[str, Union[MetricFunction, List[MetricFunction]]] = None,
    output: Union[MetricFunction, List[MetricFunction]] = None,
):
    """
    Decorator to log metrics about input and output datasets.

    Args:
        inputs: Dictionary mapping input argument names to metric functions
        output: Metric function(s) to apply to the function output

    Returns:
        Decorated function that logs metrics before and after execution

    Example:
        @log_metrics(
            inputs={
                "input_df": [log_count, log_null_count],
                "reference_data": log_count
            },
            output=log_distribution
        )
        def process_data(input_df, reference_data):
            # Function implementation
            pass
    """
    inputs = inputs or {}
    output_funcs = output if isinstance(output, list) else [output] if output else []

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function argument names and values
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            context = try_get_node_name() or func.__name__

            # Log input metrics
            for arg_name, metric_funcs in inputs.items():
                if arg_name in bound_args.arguments:
                    funcs = metric_funcs if isinstance(metric_funcs, list) else [metric_funcs]
                    for metric_func in funcs:
                        if metric_func:
                            metric_func(context, arg_name, bound_args.arguments[arg_name])

            # Execute function
            result = func(*args, **kwargs)

            # Log output metrics
            for metric_func in output_funcs:
                if metric_func:
                    metric_func(context, "output", result)

            return result

        return wrapper

    return decorator


def try_get_node_name() -> str | None:
    """Attempt to get the current Kedro node name from the call stack.

    Returns:
        Node name if found, None otherwise
    """
    try:
        frame = inspect.currentframe()
        while frame:
            if "node" in frame.f_locals and "hook_manager" in frame.f_locals:
                node = frame.f_locals["node"]
                if hasattr(node, "name"):
                    return node.name
            frame = frame.f_back
    except Exception as e:
        logger.warning(f"Unable to get node name: {e}")
        return None
