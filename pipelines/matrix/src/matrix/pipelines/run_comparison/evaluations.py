import abc

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


class ComparisonEvaluation(abc.ABC):
    """Abstract base class for run-comparison evaluations."""

    @abc.abstractmethod
    def evaluate_single_fold(self, input_matrices: dict[str, dict[int, pl.LazyFrame]]) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate_multi_fold(self, input_matrices: dict[str, dict[int, pl.LazyFrame]]) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate_bootstrap_single_fold(self, input_matrices: dict[str, dict[int, pl.LazyFrame]]) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate_bootstrap_multi_fold(self, input_matrices: dict[str, dict[int, pl.LazyFrame]]) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def plot_results(results: pl.DataFrame) -> plt.Figure:
        pass


class ComparisonEvaluationModelSpecific(ComparisonEvaluation):
    """Abstract base class for evaluations that produce a single curve for each model (e.g. recall@n, AUPRC, etc. )."""

    @abc.abstractmethod
    def give_x_values(self) -> np.ndarray:
        """Give the common x-values for all curves.

        Returns:
            A 1D numpy array of x-values.
        """
        pass

    @abc.abstractmethod
    def give_y_values(self, matrix: pl.DataFrame, score_col_name: str) -> np.ndarray:
        """Give the y-values of the curve for a single set of predictions.

        Returns:
            A 1D numpy array of y-values.
        """
        pass

    @abc.abstractmethod
    def give_y_values_bootstrap(self, matrix: pl.DataFrame, score_col_name: str) -> np.ndarray:
        """Give the y-values of all curves for each bootstrap sample.

        Returns:
            A 2D numpy array of y-values with shape (number of bootstrap samples, number of x-values).
        """
        pass

    @abc.abstractmethod
    def give_y_values_random_classifier(self, matrix: pl.DataFrame, score_col_name: str) -> np.ndarray:
        """Give the y-values of the curve for a random classifier.

        Returns:
            A 1D numpy array of y-values.
        """
        pass

    def evaluate_single_fold(self, input_matrices: dict[str, any]) -> pl.DataFrame:
        """Compute evaluation curves for all models without any uncertainty estimation.

        Args:
            input_matrices: Dictionary containing model predictions as Polars LazyFrames and score column name for each model and fold.

        Returns:
            Polars DataFrame with columns `x` and `y_{model_name}` for each model.
        """
        x_lst = self.give_x_values()
        output_dataframe = pl.DataFrame({"x": x_lst})

        for model_name, model_data in input_matrices.items():
            # Take fixed fold and materialize in memory as Polars dataframe
            matrix = model_data[0]["predictions"].collect()

            # Compute y-values and join to output results dataframe
            score_col_name = model_data[0]["score_col_name"]
            y_values = self.give_y_values(matrix, score_col_name)
            results_df_model = pl.DataFrame({"x": x_lst, f"y_{model_name}": y_values})
            output_dataframe = output_dataframe.join(results_df_model, how="left", on="x")

        return output_dataframe

    def evaluate_multi_fold(
        self,
        input_matrices: dict[str, any],
    ) -> pl.DataFrame:
        """Compute evaluation curves for all models with multi fold uncertainty estimation.

        Args:
            input_matrices: Dictionary containing model predictions as Polars LazyFrames and score column name for each model and fold.

        Returns:
            Polars DataFrame with columns `x` and `y_{model_name}_mean` and `y_{model_name}_std` for each model.
        """
        x_lst = self.give_x_values()
        output_dataframe = pl.DataFrame({"x": x_lst})

        for model_name, model_data in input_matrices.items():
            y_values_all_folds = []
            for fold, data in model_data.items():
                # Materialize predictions in memory as Polars dataframe
                matrix = data["predictions"].collect()

                # Compute y-values for fold and append to list
                score_col_name = data["score_col_name"]
                y_values_all_folds.append(self.give_y_values(matrix, score_col_name))

            # Take mean and std of y-values across folds
            y_values_all_folds = np.mean(y_values_all_folds, axis=0)
            y_values_all_folds_std = np.std(y_values_all_folds, axis=0)
            results_df_model = pl.DataFrame(
                {"x": x_lst, f"y_{model_name}_mean": y_values_all_folds, f"y_{model_name}_std": y_values_all_folds_std}
            )
            output_dataframe = output_dataframe.join(results_df_model, how="left", on="x")

        return output_dataframe

    def evaluate_bootstrap_single_fold(self, input_matrices: dict[str, any]) -> pl.DataFrame:
        """Compute evaluation curves for a single fold of models with bootstrap uncertainty estimation.

        Args:
            input_matrices: Dictionary containing model predictions as Polars LazyFrames and score column name for each model and fold.

        Returns:
            Polars DataFrame with columns `x` and `y_{model_name}_mean` and `y_{model_name}_std` for each model.
        """
        x_lst = self.give_x_values()
        output_dataframe = pl.DataFrame({"x": x_lst})

        for model_name, model_data in input_matrices.items():
            # Take fixed fold and materialize in memory as Polars dataframe
            matrix = model_data[0]["predictions"].collect()

            # Compute y-values for bootstrap then take mean and std.
            score_col_name = model_data[0]["score_col_name"]
            y_values_all_bootstraps = self.give_y_values_bootstrap(matrix, score_col_name)
            y_values_mean = np.mean(y_values_all_bootstraps, axis=0)
            y_values_std = np.std(y_values_all_bootstraps, axis=0)

            # Join to output results dataframe
            results_df_model = pl.DataFrame(
                {"x": x_lst, f"y_{model_name}_mean": y_values_mean, f"y_{model_name}_std": y_values_std}
            )
            output_dataframe = output_dataframe.join(results_df_model, how="left", on="x")

        return output_dataframe

    def evaluate_bootstrap_multi_fold(self, input_matrices: dict[str, any]) -> pl.DataFrame:
        """Compute evaluation curves for all models with both multi fold and bootstrap uncertainty estimation.

        Args:
            input_matrices: Dictionary containing model predictions as Polars LazyFrames and score column name for each model and fold.

        Returns:
            Polars DataFrame with columns `x` and `y_{model_name}_mean` and `y_{model_name}_std` for each model.
        """
        x_lst = self.give_x_values()
        output_dataframe = pl.DataFrame({"x": x_lst})

        for model_name, model_data in input_matrices.items():
            y_values_all_folds = []
            for fold, data in model_data.items():
                # Materialize predictions in memory as Polars dataframe
                matrix = data["predictions"].collect()

                # Compute y-values for bootstrap and append to list
                score_col_name = data["score_col_name"]
                y_values_all_folds.append(self.give_y_values_bootstrap(matrix, score_col_name))

            # Stack bootstrap values for different folds then take mean and std over all
            y_values_all_folds = np.vstack(y_values_all_folds)
            y_values_mean = np.mean(y_values_all_folds, axis=0)
            y_values_std = np.std(y_values_all_folds, axis=0)
            results_df_model = pl.DataFrame(
                {"x": x_lst, f"y_{model_name}_mean": y_values_mean, f"y_{model_name}_std": y_values_std}
            )
            output_dataframe = output_dataframe.join(results_df_model, how="left", on="x")

        return output_dataframe

    def plot_results(self, results: pl.DataFrame) -> plt.Figure:
        """Plot the results."""
        return  # TODO


class FullMatrixRecallAtN(ComparisonEvaluationModelSpecific):
    """Recall@N evaluation"""

    def __init__(
        self,
        bool_test_col: str,
        n_max: int,
        perform_sort: bool,
        num_n_values: int = 1000,
        N_bootstraps: int = 100,
        seed_bootstraps: int = 42,
    ):
        """Initialize an instance of FullMatrixRecallAtN.

        Args:
            bool_test_col: Boolean column in the matrix indicating the known positive test set.
            n_max: Maximum value of n to compute recall@n score for.
            perform_sort: Whether to sort the matrix or expect the dataframe to be sorted already.
            num_n_values: Number of n values to compute recall@n score for.
            N_bootstraps: Number of bootstrap samples to compute.
            seed_bootstraps: Seed for the bootstrap samples.
        """
        self.bool_test_col = bool_test_col
        self.n_max = n_max
        self.perform_sort = perform_sort
        self.num_n_values = num_n_values
        self.N_bootstraps = N_bootstraps
        self.seed_bootstraps = seed_bootstraps

    def give_x_values(self) -> np.ndarray:
        """Integer x-axis values from 0 to n_max, representing the n in recall@n."""
        return np.arange(0, self.n_max, self.num_n_values)

    def _give_ranks_series(self, matrix: pl.DataFrame, score_col_name: str) -> pl.Series:
        """Give the ranks of the known positives."""
        # Sort by score
        if self.perform_sort:
            matrix = matrix.sort(by=score_col_name, descending=True)

        return matrix.with_row_index("index").filter(pl.col(self.bool_test_col)).select(pl.col("index")).to_series() + 1

    def give_y_values(self, matrix: pl.DataFrame, score_col_name: str) -> np.ndarray:
        """Compute Recall@n values for a single set of predictions."""
        n_lst = self.give_x_values()

        # Number of known positives
        N = len(matrix.filter(pl.col(self.bool_test_col)))
        if N == 0:
            raise ValueError("No known positives in the matrix.")

        # Return Recall@n values
        ranks_series = self._give_ranks_series(matrix, score_col_name)
        return np.array([(ranks_series <= n).sum() / N for n in n_lst])

    def give_y_values_bootstrap(self, matrix: pl.DataFrame, score_col_name: str) -> np.ndarray:
        """Compute Recall@n values for all bootstrap samples of a single set of predictions."""
        n_lst = self.give_x_values()

        # Number of known positives
        N = len(matrix.filter(pl.col(self.bool_test_col)))
        if N == 0:
            return np.zeros((self.N_bootstraps, len(n_lst)))

        # Computed ranks of the known positives
        ranks_series = self._give_ranks_series(matrix, score_col_name)

        # Function for resampling and calculating recall@n
        def bootstrap_recall_lst(ranks_series, N, n_lst):
            ranks_series_resampled = ranks_series.sample(N, with_replacement=True, seed=self.seed_bootstraps).sort(
                descending=False
            )
            return [(ranks_series_resampled <= n).sum() / N for n in n_lst]

        # Calculate recall@n for each bootstrap
        return np.array([bootstrap_recall_lst(ranks_series, N, n_lst) for _ in range(self.N_bootstraps)])

    def give_y_values_random_classifier(self, matrix: pl.DataFrame, score_col_name: str) -> np.ndarray:
        """Compute Recall@n values for a random classifier."""
        return  # TODO
