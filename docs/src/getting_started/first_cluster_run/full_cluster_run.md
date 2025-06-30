---
title: Full Pipeline Run on Cluster
---
# Running the Full Matrix Pipeline on Cluster

!!! warning
    Cluster runs are resource-intensive and can take 16+ hours to complete. Ensure you have proper resource allocation and monitoring set up.

This guide covers running the complete Matrix pipeline on the Kubernetes cluster using Argo Workflows. Similarly to running the pipeline locally / on a single machine, we have several options for running the pipeline. We can run it from a specific release OR run it e2e starting from raw data. Either way, we rely on our custom `kedro experiment` functionality for submitting the runs to the cluster

### Kedro Experiment

The `kedro experiment` command is our custom extension that enables running pipelines on the Kubernetes cluster with experiment tracking. When you run an experiment:

1. It creates or uses an existing MLFlow experiment with the provided name
2. Containerizes your code and pushes it to GCP Artifact Registry
3. Creates an Argo Workflow template with pipeline specs and resource requirements
4. Submits the workflow to the Kubernetes cluster for parallel execution
5. Logs all runs and parameters in MLFlow for tracking

You can find the details on `kedro experiment` and how to utilize it successfully within the [deep dive section](../deep_dive/kedro_experiment.md)



### Monitoring the workflow

You can monitor the workflow through the web interface however you can also do it through k9s or Argo CLI: 

```bash
argo watch -n <namespace> <workflow-name>
```
### Accessing the Outputs

Similarly to the full run on your local instance, the data products are available within the `data` directory. However, after running the pipeline in the cloud environmentm, the data directory will be present in the remote GCS bucket as specified in `globals.yaml` and `{pipeline_stage}/catalog.yaml` parameters. 

When running the pipeline in the cloud environment, you can also examine all the parameters you used and (potentially) model performance metrics at the [MLFlow page](https://mlflow.platform.dev.everycure.org).

[Deep Dive :material-skip-previous:](../deep_dive/index.md){ .md-button .md-button--secondary }

