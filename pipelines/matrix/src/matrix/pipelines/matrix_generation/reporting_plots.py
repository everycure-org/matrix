"""Module containing strategies for reporting plots node"""

import abc

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


class ReportingPlotGenerator(abc.ABC):
    """Class representing generators outputting plots for the matrix generation pipeline."""

    def __init__(self, name: str) -> None:
        """Initializes a ReportingPlotGenerator instance.

        Args:
            name: Name assigned to the plot (used as filename)
        """
        self.name = name

    @abc.abstractmethod
    def generate(
        self,
        sorted_matrix_df: pd.DataFrame,
    ) -> Figure:
        """Generate a plot.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix

        Returns:
            matplotlib Figure object containing the plot
        """
        pass


class SingleScoreHistogram(ReportingPlotGenerator):
    """Class for generating histograms of a score column."""

    def __init__(
        self,
        name: str,
        score_col: str,
        is_log_y_scale: bool = True,
        figsize: tuple[float] = (10, 6),
        n_bins: float = 100,
    ) -> None:
        super().__init__(name)
        self.score_col = score_col
        self.is_log_y_scale = is_log_y_scale
        self.figsize = figsize
        self.n_bins = n_bins

    def generate(
        self,
        sorted_matrix_df: pd.DataFrame,
    ) -> Figure:
        """Generate a plot.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            score_col: Name of score column to plot
            is_log_y_scale: Whether to use log scale for y-axis
            figsize: Size of the figure
            n_bins: Number of bins for the Histogram

        Returns:
            matplotlib Figure object containing the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.hist(sorted_matrix_df[self.score_col], bins=self.n_bins, edgecolor="black")
        ax.set_xlabel(self.score_col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {self.score_col}")
        ax.grid(True, alpha=0.3)
        if self.is_log_y_scale:
            ax.set_yscale("log")
        return fig


class MultiScoreHistogram(ReportingPlotGenerator):
    """Class for generating histograms for multiple score columns."""

    def __init__(
        self,
        name: str,
        score_cols_lst: list[str],
        is_log_y_scale: bool = True,
        figsize_single: tuple[float] = (5, 5),
        n_bins: float = 100,
    ) -> None:
        super().__init__(name)
        self.score_cols_lst = score_cols_lst
        self.is_log_y_scale = is_log_y_scale
        self.figsize_single = figsize_single
        self.n_bins = n_bins

    def generate(
        self,
        sorted_matrix_df: pd.DataFrame,
    ) -> Figure:
        """Generate a plot.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            score_cols_lst: List of score columns to plot
            is_log_y_scale: Whether to use log scale for y-axis
            figsize_single : Size of single subplot.
            n_bins: Number of bins for the Histogram

        Returns:
            matplotlib Figure object containing the plot
        """
        fig, ax = plt.subplots(
            1,
            len(self.score_cols_lst),
            figsize=(self.figsize_single[0] * len(self.score_cols_lst), self.figsize_single[0]),
        )
        for j, score_col in enumerate(self.score_cols_lst):
            ax[j].hist(sorted_matrix_df[score_col], bins=self.n_bins, edgecolor="black")
            ax[j].set_xlabel(score_col)
            ax[j].set_ylabel("Frequency")
            ax[j].set_title(f"Distribution of {score_col}")
            ax[j].grid(True, alpha=0.3)
            if self.is_log_y_scale:
                ax[j].set_yscale("log")
        fig.tight_layout()
        return fig


class SingleScoreLinePlot(ReportingPlotGenerator):
    """Class for generating line plots of a score column."""

    def __init__(
        self,
        name: str,
        score_col: str,
        is_log_y_scale: bool = True,
        figsize: tuple[float] = (10, 6),
    ) -> None:
        super().__init__(name)
        self.score_col = score_col
        self.is_log_y_scale = is_log_y_scale
        self.figsize = figsize

    def generate(
        self,
        sorted_matrix_df: pd.DataFrame,
    ) -> Figure:
        """Generate a plot.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            score_col: Name of score column to plot
            is_log_y_scale: Whether to use log scale for y-axis

        Returns:
            matplotlib Figure object containing the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.plot(sorted_matrix_df["rank"], sorted_matrix_df[self.score_col])
        ax.set_ylabel(self.score_col)
        ax.set_xlabel("Rank")
        ax.set_title(f"Line plot of {self.score_col} against rank")
        ax.grid(True, alpha=0.3)
        if self.is_log_y_scale:
            ax.set_yscale("log")
        return fig


class SingleScoreScatterPlot(ReportingPlotGenerator):
    """Class for generating scatter plots of a score column against rank."""

    def __init__(
        self,
        name: str,
        score_col: str,
        n_sample: float = 1000000,
        figsize: tuple[float] = (10, 6),
        points_alpha: float = 0.03,
        points_s: float = 0.5,
    ) -> None:
        super().__init__(name)
        self.score_col = score_col
        self.figsize = figsize
        self.n_sample = n_sample
        self.points_alpha = points_alpha
        self.points_s = points_s

    def generate(
        self,
        sorted_matrix_df: pd.DataFrame,
    ) -> Figure:
        """Generate a plot.

        Args:
            sorted_matrix_df: DataFrame containing the sorted matrix
            score_col: Name of score column to plot
            n_sample: Number of sample pairs to take for the plot
            figsize : Size of the figure
            points_alpha : Transparency of the points
            points_s : Size of the points

        Returns:
            matplotlib Figure object containing the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        df_sample = sorted_matrix_df.sample(self.n_sample)
        ax.scatter(df_sample["rank"], df_sample[self.score_col], alpha=self.points_alpha, s=self.points_s)
        ax.set_ylabel(self.score_col)
        ax.set_xlabel("Rank")
        ax.set_title(f"Scatter plot of {self.score_col} against rank")
        ax.grid(True, alpha=0.3)
        return fig
