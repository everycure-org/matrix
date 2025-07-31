from unittest.mock import MagicMock, patch

from kedro.framework.context import KedroContext
from matrix.hooks import ReleaseInfoHooks


def test_extract_release_info(cloud_kedro_context: KedroContext):
    ReleaseInfoHooks.set_context(cloud_kedro_context)

    global_datasets = ReleaseInfoHooks.extract_all_global_datasets([])

    mock_mlflow = MagicMock()
    mock_mlflow.tracking.run.id = "run123"
    mock_mlflow.tracking.experiment.name = "experiment123"
    mock_mlflow.experiment_id = "555"
    cloud_kedro_context.mlflow = mock_mlflow

    expected_release_info_keys = set(
        [
            "Release Name",
            "Datasets",
            "BigQuery",
            "MLFlow",
            "Code",
            "NodeNorm Endpoint",
            "KG dashboard",
        ]
    )

    with patch("mlflow.get_experiment_by_name") as mock_mlflow:
        mock_mlflow.return_value.experiment_id = "555"
        release_info = ReleaseInfoHooks.extract_release_info(global_datasets)

        release_info_keys = set(release_info.keys())
        assert release_info_keys == expected_release_info_keys
