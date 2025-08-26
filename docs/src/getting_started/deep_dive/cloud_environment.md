---
title: Cloud Environment
--- 

# Cloud Environment Guide

The cloud environment is designed for production-scale pipeline execution on Google Cloud Platform (GCP) using our Kubernetes Cluster with Argo orchestration. It configures the MATRIX pipeline to read from and write to cloud storage and BigQuery, enabling distributed processing and scalable data operations.

The cloud environment is used for **production pipeline execution** on Kubernetes clusters, **large-scale data processing** using distributed computing, and **multi-user collaboration** with centralized data storage.

## Key Differences from Base Environment

### Data Storage Strategy

| Aspect | Base Environment | Cloud Environment |
|--------|------------------|-------------------|
| **Storage Location** | Local filesystem | GCS + BigQuery |
| **Scalability** | Limited by local resources | Cloud-scale, Parallelized |

## Data Path Structure

As mentioned, the data pathways are all pointing to our GCS buckets. You will notice that the convention remains the same between the pathways in cloud and other environments however main parent directory now points to our storage GCS bucket.
```yaml
runtime_gcs_bucket: gs://${oc.env:RUNTIME_GCP_BUCKET}
runtime_gcp_project: ${oc.env:RUNTIME_GCP_PROJECT_ID}

dev_gcs_bucket: gs://mtrx-us-central1-hub-dev-storage
prod_gcs_bucket: gs://mtrx-us-central1-hub-prod-storage
# Public GCS bucket for public datasets
public_gcs_bucket: gs://data.dev.everycure.org

# ...
paths:
  # Raw data (read-only from central buckets)
  raw: ${dev_gcs_bucket}/data/01_RAW
  raw_private: ${prod_gcs_bucket}/data/01_RAW
  # Public data sources
  raw_public: ${public_gcs_bucket}/data/01_RAW
  
  # Release-based storage
  ingestion: ${release_dir}/datasets/ingestion
  integration: ${release_dir}/datasets/integration
  release: ${release_dir}/datasets/release
  
  # Run-based storage
  filtering: ${run_dir}/datasets/filtering
  embeddings: ${run_dir}/datasets/embeddings
  modelling: ${run_dir}/datasets/modelling
  evaluation: ${run_dir}/datasets/evaluation
  matrix_generation: ${run_dir}/datasets/matrix_generation
  inference: ${run_dir}/datasets/inference
  
  # Distributed cache
  cache: ${runtime_gcs_bucket}/kedro/data/cache
```

## MLflow Cloud Configuration

The cloud environment configures MLflow for distributed execution:

```yaml
tracking:
  run: 
    # Ensures stable naming during distributed execution
    name: ${oc.env:WORKFLOW_ID}
```
We have a live MLFlow service deployed on our Kubernetes cluster where all metrics and parameters from our pipeline get stored.

## Running Cloud Environment
Runnign cloud env locally is not recommended as it relies heavily on GCS and services which are live on our cluster (e.g. live MLFlow instance). However we provide some instructions in the [cross environment](./cross_environments.md) section on how to connect to your cloud environment run and continue locally in your base env.