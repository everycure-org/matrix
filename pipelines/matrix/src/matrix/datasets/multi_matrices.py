from kedro.io.core import AbstractDataset
from kedro_datasets.polars import LazyPolarsDataset

from matrix.pipelines.run_comparison.input_paths import InputPathsMultiFold


class MultiMatricesDataset(AbstractDataset):
    """Dataset for loading multiple predictions dataframes over several models and folds.

    Different models can have different file formats but different folds must have the same format.
    """

    def __init__(self, input_paths: dict[str, InputPathsMultiFold]):
        # Lazy Polars Datasets for each fold and model.
        self._all_datasets = {
            model_name: {
                fold: LazyPolarsDataset(filepath=input_path.file_paths_list[fold], file_format=input_path.file_format)
                for fold in range(input_path.num_folds)
            }
            for model_name, input_path in input_paths.items()
        }

    def load(self):
        return {
            model_name: {fold: dataset.load() for fold, dataset in self._all_datasets[model_name].items()}
            for model_name in self._all_datasets.keys()
        }
