"""Module containing strategies for reporting plots node"""

import abc

import matplotlib.pyplot as plt
import pyspark.sql as ps
from matplotlib.figure import Figure


class ReportingPlotGenerator(abc.ABC):
    """Class representing generators outputting plots for the matrix generation pipeline."""

    def __init__(self, name: str) -> None:
        """Initializes a ReportingPlotGenerator instance.

        Args:
            name: Name assigned to the plot
        """
        self.name = name

    @abc.abstractmethod
    def generate(
        self, sorted_matrix_df: ps.DataFrame, drugs_df: ps.DataFrame, diseases_df: ps.DataFrame, **kwargs
    ) -> Figure:
        """Generate a plot.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list

        Returns:
            matplotlib Figure object containing the plot
        """
        pass


class SingleScoreHistogram(ReportingPlotGenerator):
    """Class for generating histograms of a score column."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def generate(
        self,
        sorted_matrix_df: ps.DataFrame,
        drugs_df: ps.DataFrame,
        diseases_df: ps.DataFrame,
        score_col: str,
        is_log_y_scale: bool = False,
    ) -> Figure:
        """Generate a plot.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list
            score_col: Name of score column to plot
            is_log_y_scale: Whether to use log scale for y-axis

        Returns:
            matplotlib Figure object containing the plot
        """
        return  # TODO: Implement


class MultiScoreHistogram(ReportingPlotGenerator):
    """Class for generating histograms of multiple score columns."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def generate(
        self,
        sorted_matrix_df: ps.DataFrame,
        drugs_df: ps.DataFrame,
        diseases_df: ps.DataFrame,
        score_cols_lst: list[str],
        is_log_y_scale: bool = False,
    ) -> Figure:
        """Generate a plot.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list
            score_cols_lst: List of score columns to plot
            is_log_y_scale: Whether to use log scale for y-axis

        Returns:
            matplotlib Figure object containing the plot
        """
        return  # TODO: Implement


class SingleScoreLinePlot(ReportingPlotGenerator):
    """Class for generating line plots of a score column."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def generate(
        self,
        sorted_matrix_df: ps.DataFrame,
        drugs_df: ps.DataFrame,
        diseases_df: ps.DataFrame,
        score_col: str,
        is_log_y_scale: bool = False,
    ) -> Figure:
        """Generate a plot.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            drugs_df: DataFrame containing the drugs list
            diseases_df: DataFrame containing the diseases list
            score_col: Name of score column to plot
            is_log_y_scale: Whether to use log scale for y-axis

        Returns:
            matplotlib Figure object containing the plot
        """
        return  # TODO: Implement
