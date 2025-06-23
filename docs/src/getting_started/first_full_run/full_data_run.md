---
title: Full Pipeline Run
---

# Running the Full Matrix Pipeline

Now that your Docker environment is optimized for large data processing, you can run the complete Matrix pipeline. There are two main approaches depending on your needs:

1. Full e2e run: this includes running **data engineering pipeline** first to process raw data sources to create your own data release and then using those as input to feature/modelling pipeline.
2. Using existing releases: this includes utilizing existing releases created by EC team and only running **feature & modelling pipeline**

Note that even though feature/modelling pipeline takes less time, both approaches are very time- and compute-consuming as they are processing very large datasets. When parallelizing the pipeline on our kubernetes cluster, we can complete e2e run in less than 16 hours however when limited to a single instance, you can expect it to run for more than 24 hours.

## Option 1: Full e2e run

The first part is to run the data engineering pipeline with desired data sources. You can customize which sources to include by modifying the configuration:

### 1. Configure Data Sources in Settings

Update your `conf/base/globals.yml` to specify which versions of data sources to include and `settings.py` to specify which datasets should be processed:

```yaml
# conf/base/globals.yml
data_sources:
  rtx_kg2:
    version: v2.10.0_validated
  robokop:
    version: 30fd1bfc18cd5ccb
  spoke:
    version: V5.2
  embiology:
    version: 03032025
  ec_medical_team:
    version: 20241031
  ec_clinical_trials:
    version: 20230309
  drug_list:
    version: v0.1.1
  disease_list:
    version: v0.1.1
  gt:
    version: v2.10.0_validated
  drugmech:
    version: 2.0.1
  off_label:
    version: v0.1
```

```python 
#
DYNAMIC_PIPELINES_MAPPING = lambda: disable_private_datasets(
    generate_dynamic_pipeline_mapping(
        {
            "cross_validation": {
                "n_cross_val_folds": 3,
            },
            "integration": [
                {"name": "rtx_kg2", "integrate_in_kg": True, "is_private": False},
                # Following is commented out to disable it
                # {"name": "robokop", "integrate_in_kg": True, "is_private": False},
                # ... other sources
                {"name": "drug_list", "integrate_in_kg": False, "has_edges": False},
                {"name": "disease_list", "integrate_in_kg": False, "has_edges": False},
            ],
```
Note that you need to enable drug list/disease list/ground truth to enable matrix generation and model evaluation.

### 2. Kick off the run
Once you disabled/enabled datasets of interest, you can kick off the Matrix pipeline simply by running
```bash
kedro run -p data_engineering -e base
```
Once the pipeline completes, you can proceed to feature/modelling steps - for further instructions, go to [Feature/Modelling Run](./full_data_run.md#step-2-featuremodelling-pipeline) section

## Option 2: Run from a specific release
The first part is modify your `.env` set up to ensure you are using the right pipeline setup.

### Step 1: Set Environment Variables

Create or modify your `.env` file:

```bash
# Set your runtime environment
RUNTIME_GCP_PROJECT_ID=mtrx-hub-dev-3of
RUNTIME_GCP_BUCKET=mtrx-us-central1-hub-dev-storage

# Set a unique run name for your run
RUN_NAME=my-full-data-run

# Set release version for output, this has to match the release version
RELEASE_VERSION=v0.7.0
RELEASE_FOLDER_NAME=releases
```

#### Step 2: Feature/Modelling pipeline

The Feature pipeline can be used to extract only subgraph of interest from the release. As mentioned in [first steps section](../first_steps/run_pipeline.md) you can optimize the parameters file to select what features/graphs you want to exclude and keep for your run.

```yaml
filtering:
  node_filters:
    filter_sources:
      _object: matrix.pipelines.filtering.filters.KeepRowsContainingFilter
      column: upstream_data_source
      keep_list:
        - rtxkg2
        - ec_medical
        # - robokop  # Uncomment to include ROBOKOP data
  # ...
  edge_filters:
    filter_sources:
      _object: matrix.pipelines.filtering.filters.KeepRowsContainingFilter
      column: upstream_data_source
      keep_list:
        - rtxkg2
        - ec_medical
        # - robokop

```

You might also want to optimize the embedding parameters
```yaml
embeddings.topological_estimator:
  _object: matrix.pipelines.embeddings.graph_algorithms.GDSNode2Vec 
  concurrency: 4
  embedding_dim: 512
  random_seed: 42
  iterations: 10
  walk_length: 30
  walks_per_node: 10
  window_size: 10
```


Note that Feature pipeline at the moment also relies on neo4j instance with a lot of memory. Make sure that your docker instance of neo4j has appropriate amount of memory allocated (as we have specified in cloud parameters)

TODO: neo4j 

Once you have that ready, you can run
```bash
# Run after data engineering runs to completion
kedro run -e base -p feature
```

After completing the feature extraction step, you should be ready to kick off modelling run. As mentioned in [first steps section](../first_steps/run_pipeline.md), make sure you select classifier and train-test-split of interest: by default we use stratified randomized train-test-split with an ensemble XGBoost model.

Once you are ready, you can kick off:
```bash
# Run after data engineering runs to completion
kedro run -e base -p modelling_run
```
### Pipeline Output

After successful completion, you'll have intermediate data products saved in your `data` directory with each pipeline having its own directory with intermediate data products.
