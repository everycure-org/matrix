---
title: Base Environment
--- 

# Base Environment Guide

The base environment serves as the foundation configuration for any kedro project. It contains shared configuration that is common across all environments and defines the core data paths, data source versions, and default settings. To get a good understanding of the pipeline, you need to have a good understanding of the base environment.

## Key Configuration Files

The base `globals.yml` contains essential configuration that other environments inherit and potentially override. It includes versions of the KG that should be used, GCS buckets, and paths that are then used throughout the data catalog:

```yaml
# Core data source versions
data_sources:
  rtx_kg2:
    version: v2.10.0_validated
  robokop:
    version: 30fd1bfc18cd5ccb
  spoke:
    version: V5.2
  # ... other data sources

# Default GCS bucket configuration
dev_gcs_bucket: gs://mtrx-us-central1-hub-dev-storage
prod_gcs_bucket: gs://mtrx-us-central1-hub-prod-storage
# Public GCS bucket for public datasets
public_gcs_bucket: gs://data.dev.everycure.org

# Environment variable defaults for local development
neo4j:
  host: ${oc.env:NEO4J_HOST,bolt://127.0.0.1:7687}
  user: ${oc.env:NEO4J_USER,neo4j}
  password: ${oc.env:NEO4J_PASSWORD,admin}
```

## Data Catalog Pathways

As you probably noticed, the base catalog is almost fully local. The only non-local dependency is reading the raw data which is stored in our GCS bucket - all other intermediate and final data products are stored within respective directories in the `data` directory. This means that you can easily reproduce base matrix pipeline on your local system.

## Data Path Structure

The matrix project uses different data paths depending on the type and accessibility of the data:

### Raw Data Paths

- **`kg_raw`**: Points to the dev bucket (`gs://mtrx-us-central1-hub-dev-storage/data/01_RAW/`) for general raw KG data
- **`raw_private`**: Points to the prod bucket (`gs://mtrx-us-central1-hub-prod-storage/data/01_RAW/`) for private datasets (e.g., SPOKE, Embiology)
- **`raw_public`**: Points to the public bucket (`gs://data.dev.everycure.org/data/01_RAW/`) for publicly accessible data sources (e.g., RTX-KG2, Robokop, Ground Truth)

This separation ensures that:
- Public data sources are available through the public data zone
- Private datasets remain secure in the production environment
- Development datasets can be accessed without production credentials

### Path Configuration

All paths are configured in `globals.yml` and referenced throughout the data catalog using template variables like `${globals:paths.raw_public}`.

## MLFlow and Spark

You will also notice `mlflow.yaml` and `spark.yaml` files in the config. 

MLFlow is a common experiment tracking tool for registering metrics and parameters for specific runs. MLFlow dependency is disabled in the base environment and we only use it for `cloud` environment for experiment tracking purposes. Whilst it is possible to enable it through `.env` file, we do not recommend it as there are potential issues when switching between different environments (hence we limit it only to cloud)

Spark configuration file contains parameters for optimizing your spark JVM system which is required for any environment.