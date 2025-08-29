from matrix_mlflow_utils.mlflow import MlflowMetricsDataset


def test_dataset():
    dataset = MlflowMetricsDataset()
    assert dataset is not None
