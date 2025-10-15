"""A module containing classes for processing user for the run comparison pipeline.

TODO: Add unit tests.
"""

from dataclasses import dataclass


@dataclass
class InputPathsMultiFold:
    """Class for inputting multi-fold matrix predictions."""

    name: str
    file_paths_list: list[str]
    score_col_name: str
    file_format: str = "parquet"


class InputPathSingleFold(InputPathsMultiFold):
    """Class for inputting a single fold of matrix predictions."""

    def __init__(self, name: str, file_path: str, score_col_name: str, file_format: str = "parquet"):
        super().__init__(
            name=name,
            file_paths_list=[file_path],
            score_col_name=score_col_name,
            file_format=file_format,
        )


class InputPathsModellingRun(InputPathsMultiFold):
    """Class for inputting matrices from a MATRIX modelling run."""

    def __init__(self, name: str, data_release: str, run_name: str, num_folds: int, is_transformed: bool = True):
        base_path = f"gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/{data_release}/runs/{run_name}/datasets"

        if is_transformed:
            score_col_name = "transformed_treat_score"
            file_paths_list = [
                f"{base_path}/matrix_transformations/fold_{fold}/transformed_matrix" for fold in range(num_folds)
            ]
        else:
            score_col_name = "treat score"
            file_paths_list = [f"{base_path}/model_output/fold_{fold}/matrix_predictions" for fold in range(num_folds)]

        super().__init__(
            name=name,
            file_paths_list=file_paths_list,
            score_col_name=score_col_name,
            file_format="parquet",
        )
