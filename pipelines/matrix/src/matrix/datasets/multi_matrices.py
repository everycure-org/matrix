from kedro_datasets.polars import LazyPolarsDataset
from kedro_datasets.yaml import YAMLDataset


# TODO: I think in fact we should subclass this from LazyPolarsDataset and copy all matrices to the pipeline data folder. This will substantially reduce i/o time (download only once).
class MultiMatricesDataset(YAMLDataset):
    """Dataset for loading multiple predictions dataframes over several models and folds."""

    def load(self):
        """Load a lazy Polars datasets and score column name for each fold and model."""
        paths_dict = super().load()
        return {
            model_name: {
                fold: {
                    "predictions": LazyPolarsDataset(
                        filepath=dict_for_model["file_paths_list"][fold],
                        file_format=dict_for_model["file_format"],
                    ).load(),
                    "score_col_name": dict_for_model["score_col_name"],
                }
                for fold in range(len(dict_for_model["file_paths_list"]))
            }
            for model_name, dict_for_model in paths_dict.items()
        }
