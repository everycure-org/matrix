---
title: Base Environment
--- 

# Base Environment Guide

The base environment serves as the foundation configuration for the MATRIX pipeline. It contains shared configuration that is common across all environments and defines the core data paths, data source versions, and default settings.

## Purpose

The base environment is designed to:

- Provide a common foundation for all other environments
- Define shared data source versions and configurations
- Establish default data paths and storage locations
- Configure core services like Neo4j and MLflow
- Set up environment variable defaults for local development

## Key Configuration Files

### `globals.yml`

The base `globals.yml` contains essential configuration that other environments inherit and potentially override:

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

# Environment variable defaults for local development
neo4j:
  host: ${oc.env:NEO4J_HOST,bolt://127.0.0.1:7687}
  user: ${oc.env:NEO4J_USER,neo4j}
  password: ${oc.env:NEO4J_PASSWORD,admin}
```

### Data Paths Structure

The base environment defines a hierarchical data path structure:

```yaml
paths:
  # Raw data from GCS
  raw: ${dev_gcs_bucket}/kedro/data/01_raw
  kg_raw: ${dev_gcs_bucket}/data/01_RAW
  
  # Release-based paths
  integration: data/releases/${versions.release}/datasets/integration
  release: data/releases/${versions.release}/datasets/release
  
  # Run-based paths (for pipeline outputs)
  filtering: data/releases/${versions.release}/runs/${run_name}/datasets/filtering
  embeddings: data/releases/${versions.release}/runs/${run_name}/datasets/embeddings
  modelling: data/releases/${versions.release}/runs/${run_name}/datasets/modelling
  # ... other pipeline stages
```

## Data Catalog Configuration

The base environment's data catalog (`conf/base/`) defines how datasets are loaded and saved:

### Dataset Types

- **Pandas datasets**: For smaller datasets processed locally
- **Spark datasets**: For large-scale data processing
- **GCS datasets**: For cloud storage integration
- **SQL datasets**: For database interactions

### Example Catalog Entry

```yaml
ingestion.raw.rtx_kg2.nodes@pandas:
  type: pandas.CSVDataset
  filepath: ${globals:paths.kg_raw}/KGs/rtx_kg2/${globals:data_sources.rtx_kg2.version}/rtx-kg2_2.10.0_nodes_v2-2.tsv
  load_args:
    sep: "\t"
  save_args:
    header: true
    index: false
    sep: "\t"
```

## MLflow Configuration

The base MLflow configuration sets up experiment tracking:

```yaml
server:
  mlflow_tracking_uri: ${oc.env:MLFLOW_ENDPOINT,http://127.0.0.1:5001}

tracking:
  experiment:
    name: ${globals:run_name}
    restore_if_deleted: True
  run:
    name: ${oc.env:RUN_NAME, test-run, True}
    nested: True
```

## Environment Variable Defaults

The base environment provides sensible defaults for local development:

- **Neo4j**: Defaults to localhost with standard credentials
- **GCS buckets**: Points to development storage
- **MLflow**: Defaults to local tracking server
- **Run names**: Provides fallback values for testing

## Usage Patterns

### Local Development

When developing locally, the base environment provides:

1. **Local file system storage** for intermediate datasets
2. **Default service configurations** for Neo4j and MLflow
3. **Development GCS bucket access** for raw data ingestion
4. **Flexible run naming** for testing and experimentation

### Environment Inheritance

Other environments inherit from base and override specific configurations:

- **Cloud environment**: Overrides paths to use GCS/BigQuery
- **Test environment**: Overrides parameters for faster testing
- **Sample environment**: Overrides data sources to use smaller datasets

## Best Practices

1. **Keep base configuration generic**: Avoid environment-specific settings
2. **Use environment variables**: Provide defaults but allow overrides
3. **Document data source versions**: Keep them centralized in base
4. **Maintain backward compatibility**: Changes affect all environments

## Troubleshooting

### Common Issues

1. **Missing environment variables**: Check that required variables are set in your `.env` file
2. **GCS access issues**: Ensure proper authentication for cloud storage
3. **Neo4j connection problems**: Verify Neo4j is running and accessible
4. **MLflow tracking errors**: Check MLflow server is running locally

### Debugging Commands

```bash
# Check environment configuration
kedro info

# Validate catalog configuration
kedro catalog list

# Test data loading
kedro run --node=ingestion.raw.rtx_kg2.nodes@pandas
```

The base environment provides the foundation for all MATRIX pipeline operations, ensuring consistency across different deployment scenarios while maintaining flexibility for environment-specific requirements.