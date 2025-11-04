import abc
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


class ComparisonEvaluation(abc.ABC):
    """Abstract base class for run-comparison evaluations."""

    def evaluate(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Evaluate the results.

        Args:
            combined_predictions: Dictionary of PartitionedDataset load fn's returning predictions for all folds and models
            predictions_info: Dictionary containing model names and number of folds.

        Returns:
            Polars DataFrame with the schema expected by the plot_results method.

        """

        # TODO: Assert num fods are the same

        # this always applies multifold

        x_lst = self.give_x_values()
        output_dataframe = pl.DataFrame({"x": x_lst})

        for model_name in predictions_info["model_names"]:
            y_values_all_folds = []
            for fold in range(predictions_info["num_folds"]):
                matrix = combined_predictions[model_name + "_fold_" + str(fold)]().collect()

                # Compute y-values for fold and append to list
                y_values_all_folds.append(self.give_y_values(matrix))

            # Take mean and std of y-values across folds
            y_values_mean = np.mean(y_values_all_folds, axis=0)
            y_values_std = np.std(y_values_all_folds, axis=0)
            results_df_model = pl.DataFrame(
                {"x": x_lst, f"y_{model_name}_mean": y_values_mean, f"y_{model_name}_std": y_values_std}
            )
            output_dataframe = output_dataframe.join(results_df_model, how="left", on="x")

        return output_dataframe

    @abc.abstractmethod
    def plot_results(
        self,
        results: pl.DataFrame,
        combined_pairs: dict[str, Callable[[], pl.LazyFrame]],  # Dictionary of PartitionedDataset load fn's
        predictions_info: dict[str, any],
    ) -> plt.Figure:
        """Plot the results.

        Args:
            results: Polars DataFrame with the evaluation results (output of evaluate method).
            combined_pairs: Dictionary of PartitionedDataset load fn's returning combined matrix pairs for each fold
            predictions_info: Dictionary containing model names and number of folds.

        Returns:
            Matplotlib Figure.
        """
        pass


class BootstrapComparisonEvaluation(abc.ABC):
    """Utilities for boostrap evaluation"""

    def __init__(self, n_bootstraps: int = 100, **kwargs):
        self._n_bootstraps = n_bootstraps

    # TODO: Can we make a shared plot results here?


class ComparisonEvaluationModelSpecific(ComparisonEvaluation):
    """Abstract base class for evaluations that produce a single curve for each model (e.g. recall@n, AUPRC, etc. )."""

    def __init__(self, x_axis_label: str, y_axis_label: str, title: str, force_full_y_axis: bool = False, **kwargs):
        """Initialize an instance of ComparisonEvaluationModelSpecific.

        Args:
            x_axis_label: Label for the x-axis.
            y_axis_label: Label for the y-axis.
            title: Title of the plot.
        """
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.title = title
        self.force_full_y_axis = force_full_y_axis

    @abc.abstractmethod
    def give_x_values(self) -> np.ndarray:
        """Give the common x-values for all curves.

        Returns:
            A 1D numpy array of x-values.
        """
        pass

    @abc.abstractmethod
    def give_y_values(self, matrix: pl.DataFrame) -> np.ndarray:
        """Give the y-values of the curve for a single set of predictions.

        Args:
            matrix: Polars DataFrame containing predictions and ground truth columns for a single fold and model.
                Columns include "source", "target", "score"  plus any ground truth columns

        Returns:
            A 1D numpy array of y-values.
        """
        pass

    @abc.abstractmethod
    def give_y_values_random_classifier(self, combined_pairs: dict[str, Callable[[], pl.LazyFrame]]) -> np.ndarray:
        """Give the y-values of the curve for a random classifier, given a dictionary of reference matrices of drug-disease pairs for each fold.

        Args:
            combined_pairs: Dictionary of PartitionedDataset load fn's returning combined matrix pairs for each fold
        Returns:
            A 1D numpy array of y-values.
        """
        pass

    def _is_plot_errors(self, predictions_info: dict[str, any], perform_bootstrap: bool) -> plt.Figure:
        """Whether to plot the error bars."""
        return (predictions_info["num_folds"] > 1) or perform_bootstrap

    def plot_results(
        self,
        results: pl.DataFrame,
        combined_pairs: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
        perform_bootstrap: bool,
    ) -> plt.Figure:
        """Plot the results."""

        # List of n values for recall@n plot
        x_values = self.give_x_values()

        # Set up the figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot curves
        for model_name in predictions_info["model_names"]:
            # TODO: What is this?
            if self._is_plot_errors(predictions_info, perform_bootstrap):
                av_y_values = results[f"y_{model_name}_mean"]
                std_y_values = results[f"y_{model_name}_std"]
                ax.plot(x_values, av_y_values, label=model_name)
                ax.fill_between(x_values, av_y_values - std_y_values, av_y_values + std_y_values, alpha=0.2)
            else:
                av_y_values = results[f"y_{model_name}"]
                ax.plot(x_values, av_y_values, label=model_name)

        # Plot random classifier curve
        y_values_random = self.give_y_values_random_classifier(combined_pairs)
        ax.plot(x_values, y_values_random, "k--", label="Random classifier", alpha=0.5)

        # Configure figure
        ax.set_xlabel(self.x_axis_label)
        ax.set_ylabel(self.y_axis_label)
        if self.force_full_y_axis:
            ax.set_ylim(0, 1)
        ax.grid()
        ax.legend()
        ax.set_title(self.title)
        return fig


class FullMatrixRecallAtN(ComparisonEvaluationModelSpecific):
    """Recall@N evaluation"""

    def __init__(
        self,
        ground_truth_col: str,
        n_max: int,
        perform_sort: bool,
        title: str,
        num_n_values: int = 1000,
        force_full_y_axis: bool = True,
        **kwargs,
    ):
        """Initialize an instance of FullMatrixRecallAtN.

        Args:
            ground_truth_col: Boolean column in the matrix indicating the known positive test set.
            n_max: Maximum value of n to compute recall@n score for.
            perform_sort: Whether to sort the matrix or expect the dataframe to be sorted already.
            title: Title of the plot.
            num_n_values: Number of n values to compute recall@n score for.
            force_full_y_axis: Whether to force the y-axis to be between 0 and 1.
        """
        super().__init__(x_axis_label="n", y_axis_label="Recall@n", title=title, force_full_y_axis=force_full_y_axis)
        self.ground_truth_col = ground_truth_col
        self.n_max = n_max
        self.perform_sort = perform_sort
        self.num_n_values = num_n_values

    def give_x_values(self) -> np.ndarray:
        """Integer x-axis values from 0 to n_max, representing the n in recall@n."""
        return np.linspace(1, self.n_max, self.num_n_values)

    def _give_ranks_series(self, matrix: pl.DataFrame, score_col_name: str = "score") -> pl.Series:
        """Give the ranks of the known positives."""
        # Sort by score
        if self.perform_sort:
            matrix = matrix.sort(by=score_col_name, descending=True)

        return (
            matrix.with_row_index("index").filter(pl.col(self.ground_truth_col)).select(pl.col("index")).to_series() + 1
        )

    def give_y_values(self, matrix: pl.DataFrame, score_col_name: str = "score") -> np.ndarray:
        """Compute Recall@n values for a single set of predictions."""
        n_lst = self.give_x_values()

        # Number of known positives
        N = len(matrix.filter(pl.col(self.ground_truth_col)))
        if N == 0:
            raise ValueError("No known positives in the matrix.")

        # Return Recall@n values
        ranks_series = self._give_ranks_series(matrix, score_col_name)
        return np.array([(ranks_series <= n).sum() / N for n in n_lst])

    def give_y_values_random_classifier(self, combined_pairs: dict[str, Callable[[], pl.LazyFrame]]) -> np.ndarray:
        """Compute Recall@n values for a random classifier."""
        matrix = list(combined_pairs.values())[0]()
        matrix_length = matrix.select(pl.len()).collect().item()
        return np.array([x / matrix_length for x in self.give_x_values()])


class FullMatrixRecallAtNBootstrap(FullMatrixRecallAtN, BootstrapComparisonEvaluation):
    def __init__(
        self,
        ground_truth_col: str,
        n_max: int,
        perform_sort: bool,
        title: str,
        num_n_values: int = 1000,
        force_full_y_axis: bool = True,
        n_bootstraps: int = 100,
    ):
        super().__init__(
            ground_truth_col=ground_truth_col,
            n_max=n_max,
            perform_sort=perform_sort,
            title=title,
            num_n_values=num_n_values,
            force_full_y_axis=force_full_y_axis,
            n_bootstraps=n_bootstraps,
        )

    def give_y_values(self, matrix: pl.DataFrame, score_col_name: str = "score") -> np.ndarray:
        """Compute Recall@n values for all bootstrap samples of a single set of predictions."""
        n_lst = self.give_x_values()

        # Number of known positives
        N = len(matrix.filter(pl.col(self.ground_truth_col)))
        if N == 0:
            return np.zeros((self.N_bootstraps, len(n_lst)))

        # Computed ranks of the known positives
        ranks_series = self._give_ranks_series(matrix, score_col_name)

        # Function for resampling and calculating recall@n
        def bootstrap_recall_lst(ranks_series, N, n_lst, seed):
            ranks_series_resampled = ranks_series.sample(N, with_replacement=True, seed=seed).sort(descending=False)
            return [(ranks_series_resampled <= n).sum() / N for n in n_lst]

        # Calculate recall@n for each bootstrap
        return np.array([bootstrap_recall_lst(ranks_series, N, n_lst, seed) for seed in range(self.N_bootstraps)])
