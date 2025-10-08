from kedro_datasets.polars import LazyPolarsDataset
from kedro_datasets.yaml import YAMLDataset


class MultiMatricesDataset(YAMLDataset):
    """Dataset for loading multiple predictions dataframes over several models and folds."""

    def load(self):
        """Load a lazy Polars datasets for each fold and model."""
        paths_dict = super().load()
        return {
            model_name: {
                fold: LazyPolarsDataset(
                    filepath=dict_for_model["file_paths_list"][fold],
                    file_format=dict_for_model["file_format"],
                ).load()
                for fold in range(len(dict_for_model["file_paths_list"]))
            }
            for model_name, dict_for_model in paths_dict.items()
        }
