"""Dynamic settings for run_comparison pipeline.

# NOTE: This file was partially generated using AI assistance.
"""

DYNAMIC_PIPELINE_MAPPING = {
    "run_comparison": {
        "inputs": {
            # Example entries. Update filepaths as needed.
            "auto-kg-release-v0-10-1-072eb02b.transformed_matrix": {
                "filepath": "gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/v0.10.1/runs/auto-kg-release-v0-10-1-072eb02b/datasets/matrix_transformations/fold_3/transformed_matrix"
            },
            "auto-kg-release-v0-10-2-fe36a9cb.transformed_matrix": {
                "filepath": "gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/v0.10.2/runs/auto-kg-release-v0-10-2-fe36a9cb/datasets/matrix_transformations/fold_3/transformed_matrix"
            },
        }
    }
}
