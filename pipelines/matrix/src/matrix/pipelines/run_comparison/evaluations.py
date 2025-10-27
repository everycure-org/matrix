import abc
from collections.abc import Callable
from itertools import combinations
from math import floor, log

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import entropy


class ComparisonEvaluation(abc.ABC):
    """Abstract base class for run-comparison evaluations."""

    @abc.abstractmethod
    def evaluate_single_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],  # Dictionary of PartitionedDataset load fn's
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate_multi_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate_bootstrap_single_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate_bootstrap_multi_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def plot_results(
        results: pl.DataFrame,
        combined_pairs: dict[str, Callable[[], pl.LazyFrame]],  # Dictionary of PartitionedDataset load fn's
        predictions_info: dict[str, any],
        is_plot_errors: bool,
    ) -> plt.Figure:
        pass


class ComparisonEvaluationModelSpecific(ComparisonEvaluation):
    """Abstract base class for evaluations that produce a single curve for each model (e.g. recall@n, AUPRC, etc. )."""

    def __init__(
        self,
        x_axis_label: str,
        y_axis_label: str,
        title: str,
        force_full_y_axis: bool = False,
        baseline_curve_name: str = "Random classifier",
    ):
        """Initialize an instance of ComparisonEvaluationModelSpecific.

        Args:
            x_axis_label: Label for the x-axis.
            y_axis_label: Label for the y-axis.
            title: Title of the plot.
            force_full_y_axis: Whether to force the y-axis to be between 0 and 1.
            baseline_curve_name: Name of the curve for baseline model (e.g. "Random classifier").
        """
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.title = title
        self.force_full_y_axis = force_full_y_axis
        self.baseline_curve_name = baseline_curve_name

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
    def give_y_values_baseline(self, combined_pairs: dict[str, Callable[[], pl.LazyFrame]]) -> np.ndarray:
        """Give the y-values of the curve for a baseline model, e.g. random classifier.

        Args:
            combined_pairs: Dictionary of PartitionedDataset load fn's returning combined matrix pairs for each fold

        Returns:
            A 1D numpy array of y-values.
        """
        pass

    def evaluate_single_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Compute evaluation curves for all models without any uncertainty estimation.

        Args:
            combined_predictions: Dictionary PartitionedDataset load fn's returning predictions for all folds and models
            predictions_info: Dictionary containing model names and number of folds.

        Returns:
            Polars DataFrame with columns "x" and "y_{model_name}" for each model.
        """
        x_lst = self.give_x_values()
        output_dataframe = pl.DataFrame({"x": x_lst})

        for model_name in predictions_info["model_names"]:
            matrix = combined_predictions[model_name + "_fold_0"]().collect()

            # Compute y-values and join to output results dataframe
            y_values = self.give_y_values(matrix)
            results_df_model = pl.DataFrame({"x": x_lst, f"y_{model_name}": y_values})
            output_dataframe = output_dataframe.join(results_df_model, how="left", on="x")

        return output_dataframe

    def evaluate_multi_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Compute evaluation curves for all models with multi fold uncertainty estimation.

        Args:
            combined_predictions: Dictionary PartitionedDataset load fn's returning predictions for all folds and models
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
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Compute evaluation curves for a single fold of models with bootstrap uncertainty estimation.

        Args:
            combined_predictions: Dictionary PartitionedDataset load fn's returning predictions for all folds and models
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
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Compute evaluation curves for all models with both multi fold and bootstrap uncertainty estimation.

        Args:
            combined_predictions: Dictionary PartitionedDataset load fn's returning predictions for all folds and models
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

    def _give_is_plot_errors(self, perform_multifold: bool, perform_bootstrap: bool) -> bool:
        """Determine if error bars should be plotted."""
        return perform_multifold or perform_bootstrap

    def plot_results(
        self,
        results: pl.DataFrame,
        combined_pairs: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
        perform_multifold: bool,
        perform_bootstrap: bool,
    ) -> plt.Figure:
        """Plot the results."""
        is_plot_errors = self._give_is_plot_errors(perform_multifold, perform_bootstrap)

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
        y_values_random = self.give_y_values_baseline(combined_pairs)
        ax.plot(x_values, y_values_random, "k--", label=self.baseline_curve_name, alpha=0.5)

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
        x_values_floats = np.linspace(1, self.n_max, self.num_n_values)
        return np.round(x_values_floats).astype(int)

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

    def give_y_values_baseline(self, combined_pairs: dict[str, Callable[[], pl.LazyFrame]]) -> np.ndarray:
        """Compute Recall@n values for a random classifier."""
        matrix = list(combined_pairs.values())[0]()
        matrix_length = matrix.select(pl.len()).collect().item()
        return np.array([min([1, x / matrix_length]) for x in self.give_x_values()])


class SpecificHitAtK(ComparisonEvaluationModelSpecific):
    """Drug or disease-specific Hit@k evaluation.

    Note: This version of Hit@k is defined as the proportion of known positives that are in the top k when ranked
    against all known negatives and unknown pairs with the same drug or disease (respectively).
    """

    def __init__(
        self,
        ground_truth_col: str,
        k_max: int,
        title: str,
        specific_col: str = "target",
        N_bootstraps: int = 100,
        force_full_y_axis: bool = True,
    ):
        """Initialize an instance of SpecificHitAtK.

        Args:
            ground_truth_col: Boolean column in the matrix indicating the known positive test set.
            k_max: Maximum value of k to compute Hit@k score for.
            title: Title of the plot.
            specific_col: Column to rank over.
                Set to "source" for drug-specific ranking.
                Set to "target" for disease-specific ranking.
            N_bootstraps: Number of bootstrap samples to compute.
            force_full_y_axis: Whether to force the y-axis to be between 0 and 1.
        """
        super().__init__(x_axis_label="n", y_axis_label="Recall@n", title=title, force_full_y_axis=force_full_y_axis)
        self.ground_truth_col = ground_truth_col
        self.k_max = k_max
        if specific_col not in ["source", "target"]:
            raise ValueError("specific_col must be either 'source' or 'target'.")
        self.specific_col = specific_col
        self.N_bootstraps = N_bootstraps

    def give_x_values(self) -> np.ndarray:
        """Integer x-axis values from 0 to k_max, representing the k in Hit@k."""
        return list(range(0, self.k_max + 1))

    def _give_ranks_for_test_set(self, matrix: pl.DataFrame, score_col_name: str = "score") -> pl.DataFrame:
        """Compute specific ranks against negatives and unknowns for the test set."""
        ## NOTE: Comments written for disease-specific ranking but same logic applies for drug-specific ranking.
        # Restrict to test diseases
        test_diseases = (
            matrix.group_by(self.specific_col)
            .agg(pl.col(self.ground_truth_col).sum().alias("num_known_positives"))
            .filter(pl.col("num_known_positives") > 0)
            .select(pl.col(self.specific_col))
            .to_series()
            .to_list()
        )
        matrix = matrix.filter(pl.col(self.specific_col).is_in(test_diseases))

        # Add disease-specific ranks
        matrix = matrix.with_columns(
            specific_rank=pl.col(score_col_name).rank(descending=True, method="random").over(self.specific_col)
        )

        # Return dataframe containing specific ranks against negatives and unknowns for the test set
        return (
            matrix
            # Restrict to test pairs inly
            .filter(pl.col(self.ground_truth_col))
            # Compute rank against negatives/unknowns only by taking away positives from the rank
            .with_columns(
                specific_rank_among_positives=pl.col(score_col_name)
                .rank(descending=True, method="dense")
                .over(self.specific_col)
            )
            .with_columns(
                specific_rank_against_negatives=pl.col("specific_rank") - pl.col("specific_rank_among_positives") + 1
            )
        )

    def _give_hit_at_k(self, ranks_for_test_set: pl.DataFrame, N_test_pairs: int) -> pl.DataFrame:
        """Compute a numpy array of Hit@k values for k up to k_max, given specific ranks for the test set.

        Args:
            ranks_for_test_set: Polars DataFrame containing specific rank against unknown and negatives for the test set.
                Columns must include: "specific_rank_against_negatives"
            N_test_pairs: Number of known positive test pairs.

        Returns:
            1D numpy array of Hit@k values.
        """

        # Count number of positives at each rank and cumulative sum
        num_pairs_within_rank = (
            ranks_for_test_set.group_by("specific_rank_against_negatives")
            .len()
            .sort("specific_rank_against_negatives")
            .with_columns(pl.col("len").cum_sum().alias("num_pairs_within_rank"))
        )

        # Compute hit@k for each k
        df_hit_at_k = pl.DataFrame(
            {
                "k": num_pairs_within_rank["specific_rank_against_negatives"],
                "hit_at_k": num_pairs_within_rank["num_pairs_within_rank"] / N_test_pairs,
            }
        )

        # Prepend value 0 for k=0
        df_hit_at_k = pl.concat(
            [pl.DataFrame({"k": [0], "hit_at_k": [0]}).cast({"k": pl.UInt32, "hit_at_k": pl.Float64}), df_hit_at_k]
        )

        # Join to fill missing k values
        df_hit_at_k = df_hit_at_k.join(
            pl.DataFrame({"k": self.give_x_values()}).cast(pl.UInt32), on="k", how="right"
        ).fill_null(strategy="forward")

        return df_hit_at_k["hit_at_k"].to_numpy()

    def give_y_values(self, matrix: pl.DataFrame, score_col_name: str = "score") -> np.ndarray:
        """Compute specific Hit@k values for a single set of predictions."""
        N_test_pairs = len(matrix.filter(pl.col(self.ground_truth_col)))
        ranks_for_test_set = self._give_ranks_for_test_set(matrix, score_col_name)
        return self._give_hit_at_k(ranks_for_test_set, N_test_pairs)

    def give_y_values_bootstrap(self, matrix: pl.DataFrame, score_col_name: str = "score") -> np.ndarray:
        """Compute Hit@k values for all bootstrap samples of a single set of predictions."""
        N_test_pairs = len(matrix.filter(pl.col(self.ground_truth_col)))
        ranks_for_test_set = self._give_ranks_for_test_set(matrix, score_col_name)

        # Compute Hit@k values for each bootstrap sample
        bootstrap_hit_at_k = []
        for seed in range(self.N_bootstraps):
            ranks_for_test_set_resampled = ranks_for_test_set.sample(N_test_pairs, with_replacement=True, seed=seed)
            bootstrap_hit_at_k.append(self._give_hit_at_k(ranks_for_test_set_resampled, N_test_pairs))

        # Return numpy array of Hit@k values for all bootstrap samples
        return np.array(bootstrap_hit_at_k)

    def give_y_values_baseline(self, combined_pairs: dict[str, Callable[[], pl.LazyFrame]]) -> np.ndarray:
        """Compute Hit@k values for a random classifier.

        NOTE: This is a good approximation when the number of positives per drug or disease
        is much smaller than the total number of diseases or drugs respectively.
        """
        matrix = list(combined_pairs.values())[0]()
        rank_entities_col = "source" if self.specific_col == "target" else "target"
        num_entities_per_ranking = matrix.select(rank_entities_col).unique().select(pl.len()).collect().item()
        return np.array([min([1, x / num_entities_per_ranking]) for x in self.give_x_values()])


class EntropyAtN(ComparisonEvaluationModelSpecific):
    """Drug-Entropy@N or Disease-Entropy@N evaluation."""

    def __init__(
        self,
        count_col: str,
        n_max: int,
        perform_sort: bool,
        title: str,
        num_n_values: int = 1000,
        force_full_y_axis: bool = True,
    ):
        """Initialize an instance of FullMatrixRecallAtN.

        Args:
            count_col: Column containing entities to count. Must be one of:
                "source", for Drug-Entropy@n
                "target", for Disease-Entropy@n
            n_max: Maximum value of n to compute recall@n score for.
            perform_sort: Whether to sort the matrix or expect the dataframe to be sorted already.
            title: Title of the plot.
            num_n_values: Number of n values to compute recall@n score for.
            force_full_y_axis: Whether to force the y-axis to be between 0 and 1.
        """
        super().__init__(
            x_axis_label="n",
            y_axis_label="Recall@n",
            title=title,
            force_full_y_axis=force_full_y_axis,
            baseline_curve_name="Maximum entropy",
        )
        self.count_col = count_col
        self.n_max = n_max
        self.perform_sort = perform_sort
        self.num_n_values = num_n_values

    def give_x_values(self) -> np.ndarray:
        """Integer x-axis values from 0 to n_max, representing the n in recall@n."""
        x_values_floats = np.linspace(0, self.n_max, self.num_n_values)
        return np.round(x_values_floats).astype(int)

    def give_y_values(self, matrix: pl.DataFrame, score_col_name: str = "score") -> np.ndarray:
        """Compute Drug-Entropy@n or Disease-Entropy@n values for a single set of predictions."""
        ## NOTE: Comments written for Drug-Entropy@n but same logic applies for Disease-Entropy@n.
        # Sort by treat score
        if self.perform_sort:
            matrix = matrix.sort(by=score_col_name, descending=True)

        # Total number of unique entities
        n_entities = matrix.select(pl.col(self.count_col).n_unique()).to_series().to_list()[0]
        entity_entropy_lst = [0]

        # Initialize count DataFrames with all unique entities
        entity_count = matrix.select(pl.col(self.count_col)).unique().with_columns(pl.lit(0).alias("count"))

        n_lst = self.give_x_values()
        for i in range(len(n_lst) - 1):
            # Get pairs in the new slice
            matrix_slice = matrix.slice(n_lst[i], n_lst[i + 1] - n_lst[i])

            # Count entities in the new slice
            slice_count = matrix_slice.select(pl.col(self.count_col)).to_series().value_counts(name="count_new")

            # Update total entity count
            entity_count = (
                entity_count.join(slice_count, on=self.count_col, how="left")
                .with_columns(pl.col("count_new").fill_null(0))
                .with_columns((pl.col("count") + pl.col("count_new")).alias("count"))
                .drop("count_new")
            )

            # Compute entropy
            entity_entropy_lst.append(entropy(entity_count.select("count").to_numpy().flatten(), base=n_entities))

        return entity_entropy_lst

    def give_y_values_bootstrap(self, matrix: pl.DataFrame, score_col_name: str = "score") -> np.ndarray:
        """ "Entropy@n does not use ground truth data so bootstrap uncertainty estimation is not applicable."""
        # Return same values as no bootstraps
        return self.give_y_values(matrix, score_col_name)

    def _give_maximal_entropy(self, n: int, N: int) -> float:
        """Compute maximum possible entropy value for n samples from N entities.

        NOTES:
            Let N be the number of entities and n be the number pairs (i.e the "n" in Entropy@n).
            entropy = - sum_i (p_i * log_N(p_i))
            maximal distribution has n mod N entities that appear floor(n / N) + 1 times and N - (n mod N) entities that appear floor(n / N) times.
        """
        prob_1 = (floor(n / N) + 1) / n
        prob_2 = floor(n / N) / n
        if n < N:
            return -(n % N) * prob_1 * log(prob_1, N)
        else:
            return -(n % N) * prob_1 * log(prob_1, N) - (N - (n % N)) * prob_2 * log(prob_2, N)

    def give_y_values_baseline(self, combined_pairs: dict[str, Callable[[], pl.LazyFrame]]) -> np.ndarray:
        """Compute maximum possible Entropy@n values."""
        matrix = list(combined_pairs.values())[0]().collect()
        N = len(matrix.select(pl.col(self.count_col)).unique())
        return np.array([self._give_maximal_entropy(n, N) for n in self.give_x_values()])

    def evaluate_bootstrap_single_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Override bootstrap single fold evaluation as bootstrap uncertainty estimation is not applicable."""
        return self.evaluate_single_fold(combined_predictions, predictions_info)

    def evaluate_bootstrap_multi_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Override bootstrap multi-fold evaluation as bootstrap uncertainty estimation is not applicable."""
        return self.evaluate_multi_fold(combined_predictions, predictions_info)

    def _give_plot_errors(self, perform_multifold: bool, perform_bootstrap: bool) -> bool:
        """Plot error bars if and only if multifold uncertainty estimation is performed, as bootstrap is not applicable."""
        return perform_multifold


class CommonalityAtN(ComparisonEvaluation):
    """Commonality@n evaluation for comparing similarity of pairs of predictions."""

    def __init__(
        self,
        n_max: int,
        perform_sort: bool,
        title: str,
        force_full_y_axis: bool = False,
        num_n_values: int = 1000,
    ):
        """Initialize an instance of CommonalityAtN.

        Args:
            n_max: Maximum value of n to compute commonality@n score for.
            perform_sort: Whether to sort the predictions or expect the dataframes to be sorted already.
            title: Title of the plot.
            force_full_y_axis: Whether to force the y-axis to be between 0 and 1.
            num_n_values: Number of n values to compute commonality@n score for.
        """
        self.n_max = n_max
        self.perform_sort = perform_sort
        self.title = title
        self.force_full_y_axis = force_full_y_axis
        self.num_n_values = num_n_values

    def give_n_values(self) -> np.ndarray:
        """Integer values from 0 to n_max, representing the n in commonality@n."""
        n_values = np.linspace(1, self.n_max, self.num_n_values)
        return np.round(n_values).astype(int)

    def give_commonality_values(
        self, matrix_1: pl.DataFrame, matrix_2: pl.DataFrame, score_col_name: str = "score"
    ) -> np.ndarray:
        """Give the commonality@n values for a pair of predictions.

        Args:
            matrix_1: First set of predictions.
            matrix_2: Second set of predictions.
                Both dataframes must have the columns "source, "target" and, if perform_sort is True, the column score_col_name.
        """
        if self.perform_sort:
            matrix_1 = matrix_1.sort(by=score_col_name, descending=True)
            matrix_2 = matrix_2.sort(by=score_col_name, descending=True)

        # Restrict to required rows and columns
        matrix_1 = matrix_1.select("source", "target").head(self.n_max)
        matrix_2 = matrix_2.select("source", "target").head(self.n_max)

        # Add rank columns and join to a set of common pairs
        matrix_common = (
            pl.concat([matrix_1, matrix_2])
            .unique()
            .join(matrix_1.with_row_index(name="matrix_1_rank", offset=1), on=["source", "target"], how="left")
            .join(matrix_2.with_row_index(name="matrix_2_rank", offset=1), on=["source", "target"], how="left")
        )

        # Compute number of common pairs in teh top n for each n
        n_values = self.give_n_values()
        num_common_at_n = np.array(
            [
                matrix_common.filter((pl.col("matrix_1_rank") <= n) & (pl.col("matrix_2_rank") <= n)).height
                for n in n_values
            ]
        )

        # Return proportion of pairs in common in the top n for each n
        return num_common_at_n / n_values

    def evaluate_single_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Compute evaluation curves for all pairs of models without any uncertainty estimation.

        Args:
            combined_predictions: Dictionary PartitionedDataset load fn's returning predictions for all folds and models
            predictions_info: Dictionary containing model names and number of folds.

        Returns:
            Polars DataFrame with columns "n" and "commonality_{model_name_1}_{model_name_2}" for each pair of models.
        """
        n_values = self.give_n_values()
        output_dataframe = pl.DataFrame({"n": n_values})

        for model_name_1, model_name_2 in combinations(predictions_info["model_names"], 2):
            matrix_1 = combined_predictions[model_name_1 + "_fold_0"]().collect()
            matrix_2 = combined_predictions[model_name_1 + "_fold_0"]().collect()

            # Compute y-values and join to output results dataframe
            commonality_values = self.give_commonality_values(matrix_1, matrix_2)
            results_df_pair = pl.DataFrame(
                {"n": n_values, f"commonality_{model_name_1}_{model_name_2}": commonality_values}
            )
            output_dataframe = output_dataframe.join(results_df_pair, how="left", on="n")

        return output_dataframe

    def evaluate_multi_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Compute evaluation curves for all pairs of models with multi fold uncertainty estimation.

        Args:
            combined_predictions: Dictionary PartitionedDataset load fn's returning predictions for all folds and models
            predictions_info: Dictionary containing model names and number of folds.

        Returns:
            Polars DataFrame with columns:
              - "n"
              - "commonality_{model_name_1}_{model_name_2}_mean"
              - "commonality_{model_name_1}_{model_name_2}_std"
            for each pair of models.
        """
        n_values = self.give_n_values()
        output_dataframe = pl.DataFrame({"n": n_values})

        for model_name_1, model_name_2 in combinations(predictions_info["model_names"], 2):
            commonality_values_all_folds = []
            for fold in range(predictions_info["num_folds"]):
                matrix_1 = combined_predictions[model_name_1 + "_fold_" + str(fold)]().collect()
                matrix_2 = combined_predictions[model_name_2 + "_fold_" + str(fold)]().collect()

                # Compute y-values for fold and append to list
                commonality_values_all_folds.append(self.give_commonality_values(matrix_1, matrix_2))

            # Take mean and std of y-values across folds
            commonality_values_mean = np.mean(commonality_values_all_folds, axis=0)
            commonality_values_std = np.std(commonality_values_all_folds, axis=0)
            results_df_pair = pl.DataFrame(
                {
                    "n": n_values,
                    f"commonality_{model_name_1}_{model_name_2}_mean": commonality_values_mean,
                    f"commonality_{model_name_1}_{model_name_2}_std": commonality_values_std,
                }
            )
            output_dataframe = output_dataframe.join(results_df_pair, how="left", on="n")

        return output_dataframe

    def evaluate_bootstrap_single_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Override bootstrap uncertainty estimation as it is not applicable (commonality does not use ground truth data)."""
        return self.evaluate_single_fold(combined_predictions, predictions_info)

    def evaluate_bootstrap_multi_fold(
        self,
        combined_predictions: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
    ) -> pl.DataFrame:
        """Override bootstrap uncertainty estimation as it is not applicable (commonality does not use ground truth data)."""
        return self.evaluate_multi_fold(combined_predictions, predictions_info)

    def plot_results(
        self,
        results: pl.DataFrame,
        combined_pairs: dict[str, Callable[[], pl.LazyFrame]],
        predictions_info: dict[str, any],
        perform_multifold: bool,
        perform_bootstrap: bool,
    ) -> plt.Figure:
        """Plot the results."""
        is_plot_errors = perform_multifold or perform_bootstrap

        # List of n values for recall@n plot
        n_values = self.give_n_values()

        # Set up the figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot curves
        for model_name_1, model_name_2 in combinations(predictions_info["model_names"], 2):
            if is_plot_errors:
                av_y_values = results[f"commonality_{model_name_1}_{model_name_2}_mean"]
                std_y_values = results[f"commonality_{model_name_1}_{model_name_2}_std"]
                ax.plot(n_values, av_y_values, label=f"{model_name_1} vs {model_name_2}")
                ax.fill_between(n_values, av_y_values - std_y_values, av_y_values + std_y_values, alpha=0.2)
            else:
                av_y_values = results[f"commonality_{model_name_1}_{model_name_2}"]
                ax.plot(n_values, av_y_values, label=f"{model_name_1} vs {model_name_2}")

        # Configure figure
        ax.set_xlabel("n")
        ax.set_ylabel("Commonality@n")
        if self.force_full_y_axis:
            ax.set_ylim(0, 1.05)
        ax.grid()
        ax.legend()
        ax.set_title(self.title)
        return fig
