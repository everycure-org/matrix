---
title: Cluster Specific Config 
---

# Matrix Pipeline on the Cluster

This guide covers running the complete Matrix pipeline on the Kubernetes cluster using Argo Workflows. The cluster provides distributed computing capabilities that allow for parallel processing of large datasets.

!!! warning
    Note that this section is heavily focusing on the infrastructure which can be only applicable to the Matrix Project & Matrix GCP. Therefore, this section is useful and applicable if you can access our infrastructure.
    
    If you intend to adapt Matrix Codebase & Infrastructure to your own cloud system, these instructions might be also helpful to give you an idea how we utilize the cluster however they might not be 1:1 comparable. 

## Prerequisites

Before running on the cluster, ensure you have:

1. **Cluster Access**: Completed [cluster setup](../deep_dive/cluster_setup.md)
2. **Authentication**: Valid GCP credentials and cluster access
3. **Environment Variables**: Proper `.env` configuration for your target environment
4. **Resource Understanding**: Knowledge of [Argo resource configuration](../deep_dive/kedro_extensions.md#how-to-request-resource-availability-for-a-node)

## Environment Configuration

### Required Environment Variables

Configure your `.env` file based on your target environment:

=== "Development Environment"

    ```bash
    # GCP Project and Storage
    RUNTIME_GCP_PROJECT_ID=mtrx-hub-dev-3of
    RUNTIME_GCP_BUCKET=mtrx-us-central1-hub-dev-storage
    
    # MLflow Configuration
    MLFLOW_URL=https://mlflow.platform.dev.everycure.org/
    
    # Argo Platform
    ARGO_PLATFORM_URL=https://argo.platform.dev.everycure.org
    
    # Authentication
    GOOGLE_APPLICATION_CREDENTIALS=/Users/<YOUR_USERNAME>/.config/gcloud/application_default_credentials.json
    
    # Dataset Access (development only has public datasets)
    INCLUDE_PRIVATE_DATASETS=0
    ```

=== "Production Environment"

    ```bash
    # GCP Project and Storage
    RUNTIME_GCP_PROJECT_ID=mtrx-hub-prod-sms
    RUNTIME_GCP_BUCKET=mtrx-us-central1-hub-prod-storage
    
    # MLflow Configuration
    MLFLOW_URL=https://mlflow.platform.prod.everycure.org/
    
    # Argo Platform
    ARGO_PLATFORM_URL=https://argo.platform.prod.everycure.org
    
    # Authentication
    GOOGLE_APPLICATION_CREDENTIALS=/Users/<YOUR_USERNAME>/.config/gcloud/application_default_credentials.json
    
    # Dataset Access (production includes private datasets)
    INCLUDE_PRIVATE_DATASETS=1
    ```

### Run Configuration

Set a unique run name and release version:

```bash
# Unique identifier for your run
RUN_NAME=my-full-cluster-run

# Release version for output organization
RELEASE_VERSION=v0.7.0
RELEASE_FOLDER_NAME=releases
```

## Cloud Environment Overview

The cloud environment is specifically designed for pipeline execution on GCP using our Kubernetes Cluster with Argo orchestration. Key differences from the base environment include:

- **Storage Strategy**: Uses GCS buckets and BigQuery instead of local filesystem
- **Scalability**: Enables cloud-scale parallel processing vs local resource limitations
- **Data Paths**: All paths point to GCS buckets following the same structure:
  ```yaml
  paths:
    raw: ${dev_gcs_bucket}/kedro/data/01_raw
    ingestion: ${release_dir}/datasets/ingestion
    integration: ${release_dir}/datasets/integration
    # ... etc
  ```
- **MLflow Integration**: Uses a live MLflow service deployed on the cluster for metrics and parameter tracking

## Argo Resource Configuration

The Matrix pipeline uses `ArgoNode` and `ArgoResourceConfig` to request specific Kubernetes resources for each pipeline step. This ensures optimal resource allocation and parallel execution.

### Default Resource Configuration

The pipeline uses these default resources per node:

```python
# Memory (GiB)
KUBERNETES_DEFAULT_LIMIT_RAM = 52
KUBERNETES_DEFAULT_REQUEST_RAM = 52

# CPU (cores)
KUBERNETES_DEFAULT_LIMIT_CPU = 14
KUBERNETES_DEFAULT_REQUEST_CPU = 4

# GPUs
KUBERNETES_DEFAULT_NUM_GPUS = 0
```

### Custom Resource Requests

For compute-intensive steps, you can specify custom resources using predefined configurations:

```python
from matrix.kedro4argo_node import ArgoNode

ArgoNode(
      func=nodes.reduce_embeddings_dimension,
      inputs={
          "df": "embeddings.feat.graph.node_embeddings@spark",
          "unpack": "params:embeddings.dimensionality_reduction",
      },
      outputs="embeddings.feat.graph.pca_node_embeddings",
      name="apply_pca",
      tags=["argowf.fuse", "argowf.fuse-group.node_embeddings"],
      argo_config=ArgoResourceConfig(
          cpu_request=14,
          cpu_limit=14,
          memory_limit=120,
          memory_request=120,
          ephemeral_storage_request=256,
          ephemeral_storage_limit=256,
      ),
  ),
```

Note that this ArgoNode is just a wrapper around a kedro node that's specifically designed for cluster runs, allowing us to granularly control resources for specific parts of the pipeline. Alternatively you can also just extract pre-existing argo node configurations:

!!! note "Fuse Tags"
    You might have noticed that, additionally to the Argo Config Resources, there are some `argo-related` tags. These tags allow to fuse kedro nodes into one argo node, meaning that series of kedro nodes uner a specific argo-tag (e.g. `argowf.fuse-group.<group_name>`) will be executed on a single machine. This is beneficial when we don't want to keep a transient intermediate data product (e.g. bucketized embeddings for parallel processing)
    


```python
from matrix.kedro4argo_node import ArgoNode, ARGO_GPU_NODE_MEDIUM

# Example: Using GPU resources for embedding computation
node(
    func=compute_embeddings,
    inputs=["processed_graph"],
    outputs=["embeddings"],
    name="compute_embeddings",
    tags=["embeddings"],
    argo_config=ARGO_GPU_NODE_MEDIUM  # Requests GPU resources
)
```

## MLflow Integration

The cluster pipeline automatically integrates with MLflow for experiment tracking and model versioning.

### MLflow Configuration

MLflow is configured through the `mlflow.yml` file in your environment configuration:

```yaml
mlflow:
  tracking_uri: ${MLFLOW_URL}
  experiment_name: ${RUN_NAME}
  artifact_root: ${gcs_bucket}/runs/${run_name}/mlflow
  registry_uri: ${MLFLOW_URL}
```

[Full Cluster Run :material-skip-previous:](../first_cluster_run/full_cluster_run.md){ .md-button .md-button--secondary }

