from matrix_gcp_datasets.gcp import BigQueryTableDataset


def test_bigquery_table_dataset():
    assert (
        BigQueryTableDataset(project_id="mtrx-hub-dev-3of", dataset="release_local-release", table="nodes") is not None
    )
