from unittest.mock import MagicMock, patch

import pytest
from kedro.framework.context import KedroContext
from matrix.hooks import ReleaseInfoHooks


@pytest.fixture
def mock_context():
    mock_context = MagicMock(spec=KedroContext)
    mock_context.config_loader = {
        "globals": {
            "versions": {"release": "1.0.0"},
            "runtime_gcp_project": "test-project",
            "data_sources": {
                "robokop": {"version": "v1"},
                "spoke": {"version": "v2"},
                "ec_medical_team": {"version": 20241031},
            },
        },
        "parameters": {
            "embeddings.topological_estimator": {"_object": "topological_estimator_v1"},
            "embeddings.node": {"resolver": {"encoder": {"model": "node_encoder_v1"}}},
            "integration": {
                "normalization": {"normalizer": {"endpoint": "https://nodenorm.transltr.io/1.5/get_normalized_nodes"}}
            },
        },
    }
    mock_mlflow = MagicMock()
    mock_mlflow.tracking.run.id = "run123"
    mock_mlflow.tracking.experiment.name = "experiment123"
    mock_context.mlflow = mock_mlflow
    return mock_context


def test_extract_release_info(mock_context):
    ReleaseInfoHooks.set_context(mock_context)

    datasets_used = ["robokop"]
    datasets_to_hide = ["spoke"]
    global_datasets = ReleaseInfoHooks.extract_all_global_datasets(datasets_to_hide)
    ReleaseInfoHooks.mark_unused_datasets(global_datasets, datasets_used)

    expected_release_info = {
        "Release Name": "1.0.0",
        "Datasets": {
            "robokop": "v1",
            "ec_medical_team": "not included",
        },
        "Topological Estimator": "topological_estimator_v1",
        "Embeddings Encoder": "node_encoder_v1",
        "BigQuery Link": (
            "https://console.cloud.google.com/bigquery?"
            "project=test-project"
            "&ws=!1m4!1m3!3m2!1smtrx-hub-dev-3of!2srelease_1_0_0"
        ),
        "MLFlow Link": ("https://mlflow.platform.dev.everycure.org/#/experiments/555/runs/run123"),
        "Code Link": "https://github.com/everycure-org/matrix/tree/1.0.0",
        "Neo4j Link": "coming soon!",
        "NodeNorm Endpoint Link": "https://nodenorm.transltr.io/1.5/get_normalized_nodes",
        "KG dashboard link": "https://data.dev.everycure.org/versions/1.0.0/evidence/",
    }

    with patch("mlflow.get_experiment_by_name") as mock_mlflow:
        mock_mlflow.return_value.experiment_id = "555"
        release_info = ReleaseInfoHooks.extract_release_info(global_datasets)
        assert release_info == expected_release_info
