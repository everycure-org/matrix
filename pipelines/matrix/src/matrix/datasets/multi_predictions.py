from kedro_datasets.polars import LazyPolarsDataset
from kedro_datasets.yaml import YAMLDataset


class MultiPredictionsDataset(YAMLDataset):
    """Dataset for loading multiple predictions dataframes over several models and folds."""

    def load(self):
        """Load a lazy Polars datasets and score column name for each fold and model."""
        data_dicts_list = super().load()
        return {
            dict_for_model["name"]: {
                "predictions_list": [
                    LazyPolarsDataset(
                        filepath=file_path,
                        file_format=dict_for_model["file_format"],
                    ).load()
                    for file_path in dict_for_model["file_paths_list"]
                ],
                "score_col_name": dict_for_model["score_col_name"],
            }
            for dict_for_model in data_dicts_list
        }
