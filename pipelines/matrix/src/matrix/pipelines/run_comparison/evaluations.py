import abc

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


class ComparisonEvaluation(abc.ABC):
    """Abstract base class for run-comparison evaluations."""

    @abc.abstractmethod
    def evaluate_single_fold(
        self,
        combined_predictions: pl.LazyFrame,
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate_multi_fold(
        self,
        combined_predictions: pl.LazyFrame,
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate_bootstrap_single_fold(
        self,
        combined_predictions: pl.LazyFrame,
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate_bootstrap_multi_fold(
        self,
        combined_predictions: pl.LazyFrame,
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def plot_results(
        results: pl.DataFrame,
        combined_predictions: pl.LazyFrame,
        predictions_info: dict[str, any],
        is_plot_errors: bool,
    ) -> plt.Figure:
        pass


class ComparisonEvaluationModelSpecific(ComparisonEvaluation):
    """Abstract base class for evaluations that produce a single curve for each model (e.g. recall@n, AUPRC, etc. )."""

    def __init__(self, x_axis_label: str, y_axis_label: str, title: str, force_full_y_axis: bool = False):
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
    def give_y_values_bootstrap(self, matrix: pl.DataFrame) -> np.ndarray:
        """Give the y-values of all curves for each bootstrap sample.

        Args:
            matrix: Polars DataFrame containing predictions and ground truth columns for a single fold and model.
                Columns include "source", "target", "score"  plus any ground truth columns

        Returns:
            A 2D numpy array of y-values with shape (number of bootstrap samples, number of x-values).
        """
        pass

    @abc.abstractmethod
    def give_y_values_random_classifier(self, combined_predictions: pl.LazyFrame) -> np.ndarray:
        """Give the y-values of the curve for a random classifier.

        Returns:
            A 1D numpy array of y-values.
        """
        pass

    def evaluate_single_fold(
        self,
        combined_predictions: pl.LazyFrame,
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Compute evaluation curves for all models without any uncertainty estimation.

        Args:
            combined_predictions: Polars LazyFrame containing predictions for all folds and models
            predictions_info: Dictionary containing model names and number of folds.

        Returns:
            Polars DataFrame with columns `x` and `y_{model_name}` for each model.
        """
        x_lst = self.give_x_values()
        output_dataframe = pl.DataFrame({"x": x_lst})

        for model_name in predictions_info["model_names"]:
            matrix = combined_predictions["model_name_" + model_name + "_fold_0"]().collect()

            # Compute y-values and join to output results dataframe
            y_values = self.give_y_values(matrix)
            results_df_model = pl.DataFrame({"x": x_lst, f"y_{model_name}": y_values})
            output_dataframe = output_dataframe.join(results_df_model, how="left", on="x")

        return output_dataframe

    def evaluate_multi_fold(
        self,
        combined_predictions: pl.LazyFrame,
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Compute evaluation curves for all models with multi fold uncertainty estimation.

        Args:
            combined_predictions: Polars LazyFrame containing predictions for all folds and models
            predictions_info: Dictionary containing model names and number of folds.

        Returns:
            Polars DataFrame with columns `x` and `y_{model_name}_mean` and `y_{model_name}_std` for each model.
        """
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

    def evaluate_bootstrap_single_fold(
        self,
        combined_predictions: pl.LazyFrame,
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Compute evaluation curves for a single fold of models with bootstrap uncertainty estimation.

        Args:
            combined_predictions: Polars LazyFrame containing predictions for all folds and models
            predictions_info: Dictionary containing model names and number of folds.

        Returns:
            Polars DataFrame with columns `x` and `y_{model_name}_mean` and `y_{model_name}_std` for each model.
        """
        x_lst = self.give_x_values()
        output_dataframe = pl.DataFrame({"x": x_lst})

        for model_name in predictions_info["model_names"]:
            matrix = combined_predictions[model_name + "_fold_0"]().collect()

            # Compute y-values for bootstrap then take mean and std.
            y_values_all_bootstraps = self.give_y_values_bootstrap(matrix)
            y_values_mean = np.mean(y_values_all_bootstraps, axis=0)
            y_values_std = np.std(y_values_all_bootstraps, axis=0)

            # Join to output results dataframe
            results_df_model = pl.DataFrame(
                {"x": x_lst, f"y_{model_name}_mean": y_values_mean, f"y_{model_name}_std": y_values_std}
            )
            output_dataframe = output_dataframe.join(results_df_model, how="left", on="x")

        return output_dataframe

    def evaluate_bootstrap_multi_fold(
        self,
        combined_predictions: dict[str, pl.LazyFrame],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Compute evaluation curves for all models with both multi fold and bootstrap uncertainty estimation.

        Args:
            combined_predictions: Polars LazyFrame containing predictions for all folds and models
            predictions_info: Dictionary containing model names and number of folds.

        Returns:
            Polars DataFrame with columns `x` and `y_{model_name}_mean` and `y_{model_name}_std` for each model.
        """
        x_lst = self.give_x_values()
        output_dataframe = pl.DataFrame({"x": x_lst})

        for model_name in predictions_info["model_names"]:
            y_values_all_folds = []
            for fold in range(predictions_info["num_folds"]):
                matrix = combined_predictions[model_name + "_fold_" + str(fold)]().collect()

                # Compute y-values for bootstrap and append to list
                y_values_all_folds.append(self.give_y_values_bootstrap(matrix))

            # Stack bootstrap values for different folds then take mean and std over all
            y_values_all_folds = np.vstack(y_values_all_folds)
            y_values_mean = np.mean(y_values_all_folds, axis=0)
            y_values_std = np.std(y_values_all_folds, axis=0)
            results_df_model = pl.DataFrame(
                {"x": x_lst, f"y_{model_name}_mean": y_values_mean, f"y_{model_name}_std": y_values_std}
            )
            output_dataframe = output_dataframe.join(results_df_model, how="left", on="x")

        return output_dataframe

    def plot_results(
        self,
        results: pl.DataFrame,
        combined_predictions: pl.LazyFrame,
        predictions_info: dict[str, any],
        is_plot_errors: bool,
    ) -> plt.Figure:
        """Plot the results."""

        # List of n values for recall@n plot
        x_values = self.give_x_values()

        # Set up the figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot curves
        for model_name in predictions_info["model_names"]:
            if is_plot_errors:
                av_y_values = results[f"y_{model_name}_mean"]
                std_y_values = results[f"y_{model_name}_std"]
                ax.plot(x_values, av_y_values, label=model_name)
                ax.fill_between(x_values, av_y_values - std_y_values, av_y_values + std_y_values, alpha=0.2)
            else:
                av_y_values = results[f"y_{model_name}"]
                ax.plot(x_values, av_y_values, label=model_name)

        # Plot random classifier curve
        y_values_random = self.give_y_values_random_classifier(combined_predictions)
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
        N_bootstraps: int = 100,
        force_full_y_axis: bool = True,
    ):
        """Initialize an instance of FullMatrixRecallAtN.

        Args:
            ground_truth_col: Boolean column in the matrix indicating the known positive test set.
            n_max: Maximum value of n to compute recall@n score for.
            perform_sort: Whether to sort the matrix or expect the dataframe to be sorted already.
            title: Title of the plot.
            num_n_values: Number of n values to compute recall@n score for.
            N_bootstraps: Number of bootstrap samples to compute.
            force_full_y_axis: Whether to force the y-axis to be between 0 and 1.
        """
        super().__init__(x_axis_label="n", y_axis_label="Recall@n", title=title, force_full_y_axis=force_full_y_axis)
        self.ground_truth_col = ground_truth_col
        self.n_max = n_max
        self.perform_sort = perform_sort
        self.num_n_values = num_n_values
        self.N_bootstraps = N_bootstraps

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

    def give_y_values_bootstrap(self, matrix: pl.DataFrame, score_col_name: str = "score") -> np.ndarray:
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

    def give_y_values_random_classifier(self, combined_predictions: pl.LazyFrame) -> np.ndarray:
        """Compute Recall@n values for a random classifier."""
        matrix = list(combined_predictions.values())[0]()
        matrix_length = matrix.select(pl.len()).collect().item()
        return np.array([x / matrix_length for x in self.give_x_values()])
