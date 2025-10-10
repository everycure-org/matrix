from kedro_datasets.polars import LazyPolarsDataset
from kedro_datasets.yaml import YAMLDataset


# TODO: I think in fact we should subclass this from LazyPolarsDataset and hence copy all matrices to the pipeline data folder. This will substantially reduce i/o time when running locally (download only once).
class MultiMatricesDataset(YAMLDataset):
    """Dataset for loading multiple predictions dataframes over several models and folds."""

    def load(self):
        """Load a lazy Polars datasets and score column name for each fold and model."""
        data_dicts_list = super().load()
        return {
            dict_for_model["name"]: {
                fold: {
                    "predictions": LazyPolarsDataset(
                        filepath=file_path,
                        file_format=dict_for_model["file_format"],
                    ).load(),
                    "score_col_name": dict_for_model["score_col_name"],
                }
                for fold, file_path in enumerate(dict_for_model["file_paths_list"])
            }
            for dict_for_model in data_dicts_list
        }
