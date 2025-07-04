---
title: Full Pipeline Run on Cluster
---
# Running the Full Matrix Pipeline on Cluster

This guide covers running the complete Matrix pipeline on the Kubernetes cluster using Argo Workflows. Similarly to running the pipeline locally / on a single machine, we have several options for running the pipeline. We can run it from a specific release OR run it e2e starting from raw data. Either way, we rely on our custom `kedro experiment` functionality for submitting the runs to the cluster

## Kedro Experiment

Similarly to how the pipeline is executed locally, we need to select what pipeline we want to run and whether it should be from scratch (_i.e._ you want to create your own data release and then run feature & modelling pipeline) or from an existing release (_i.e._ if you want to utilize an existign dataset and thus skip data_engineering pipeline). This time however you will also need to select your **run name**, **username**, and **experiment name** for tracking of the runs on the cluster. This is because currently all cluster runs are executed through `kedro experiment` functionality.

The `kedro experiment` command is our custom extension that enables running specific pipelines & releases on the Kubernetes cluster. It containerizes the codebase and pushes it to GCP Artifact Registry after which it uses Argo Workflow template with pipeline specsifications & resource requirements. The workflow is then submitted to the kubernetes cluster for parallel execution. You can find more details on `kedro experiment` and how to utilize it successfully within the [deep dive section](../deep_dive/kedro_experiment.md). Here, for simplicity, we will treat it like a black-box just for submitting the runs on the cluster. 

The experiment run command adheres to the following convention:
```bash
kedro experiment run --pipeline <pipeline_name> --username <username> --experiment-name <descriptive_name_for_parallel_experiments_tracking> --run-name <descriptive_name_for_single_run> --release-version <release_version_name> --is-test 
```

### Example run
Let's assume you (example_user123) want to run the pipeline e2e for an experiment benchmarking different biomedical knowledge graph. Similarly, you would first run the `data_engineering` pipeline:
```bash
kedro experiment run --pipeline data_engineering --username example_user123 --experiment-name benchmarking_kgs_june2025 --run-name benchmarking_kgs_reference_run --release-version v5.7-benchmark-kg-reference --is-test
```
You will be prompted to confirm an experiment name, run-name, and confirm your settings. This will then execute pipeline run on the kubernetes cluster with exactly the same codebase as is on your machine. Once the docker finishes containerization & pushing the container into GCP Artifact Registry, you should get an Argo Workflow UI link to monitor your run.

!!! tip "Monitoring the workflow"
    You can monitor the workflow through the web interface. Alternatively, you can use k9s or the Argo CLI:
    ```bash
    argo watch -n <namespace> <workflow-name>
    ```

Once it finishes and you wish to continue running the rest of the pipeline, you will need to run `feature` pipeline. Note that you will need to use _the same release version_ as it is required to pick up correct data products. Therefore, we will only change the pipeline argument:

!!! note "Use existing release"
    As mentioned in the guide for [running the pipeline locally](../first_full_run/full_data_run.md), you don't need to run the pipeline e2e; you can utilize the [existing releases](../../releases/release_history.md).

```bash
kedro experiment run --pipeline feature --username example_user123 --experiment-name benchmarking_kgs_june2025 --run-name benchmarking_kgs_reference_run --release-version v5.7-benchmark-kg-reference --is-test
```
The same process can be repeated for running `modelling_run` pipeline however here you should specify a descriptive run name as it will be encoded within the filepaths for all your data products
```bash
kedro experiment run --pipeline modelling_run --username example_user123 --experiment-name benchmarking_kgs_june2025 --run-name random_forest_benchmark_kgs_ref --release-version v5.7-benchmark-kg-reference --is-test
```

Once all these runs finish (it will take between 16-24 hrs to finish all of them using our default settings), you can check all data products on GCP by looking up `release_v5.7-benchmark-kg-reference` name (within the `test` or `release`) for parent data directory and `random_forest_benchmark_kgs_ref` name for modelling run. When running the pipeline in the cloud environment, you can also examine all the parameters you used and (potentially) model performance metrics at the [MLFlow page](https://mlflow.platform.dev.everycure.org).

[Deep Dive :material-skip-previous:](../deep_dive/index.md){ .md-button .md-button--secondary }

