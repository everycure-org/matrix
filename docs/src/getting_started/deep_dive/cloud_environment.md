---
title: Cloud Environment
--- 

# Cloud Environment Guide

The cloud environment is designed for production-scale pipeline execution on Google Cloud Platform (GCP). It configures the MATRIX pipeline to read from and write to cloud storage and BigQuery, enabling distributed processing and scalable data operations.

## Purpose

The cloud environment is optimized for:

- **Production pipeline execution** on Kubernetes clusters
- **Large-scale data processing** using distributed computing
- **Cloud-native storage** with GCS and BigQuery integration
- **Stateless operations** suitable for containerized deployments
- **Multi-user collaboration** with centralized data storage

## Key Differences from Base Environment

### Data Storage Strategy

| Aspect | Base Environment | Cloud Environment |
|--------|------------------|-------------------|
| **Storage Location** | Local filesystem | GCS + BigQuery |
| **Data Persistence** | Ephemeral | Persistent |
| **Scalability** | Limited by local resources | Cloud-scale |
| **Collaboration** | Single user | Multi-user |

### Configuration Overrides

The cloud environment overrides several key configurations from the base environment:

```yaml
# Cloud-specific globals.yml overrides
run_name: ${oc.env:RUN_NAME}
versions:
  release: ${oc.env:RELEASE_VERSION,v0.6.0}

# Cloud storage paths
release_dir: ${runtime_gcs_bucket}/kedro/data/${release_folder_name}/${versions.release}
run_dir: ${release_dir}/runs/${run_name}

# MLflow configuration for cloud
mlflow_artifact_root: ${run_dir}/mlflow
mlflow_experiment_id: ${oc.env:MLFLOW_EXPERIMENT_ID, 1}
```

## Cloud Infrastructure Components

### Google Cloud Storage (GCS)

The cloud environment uses GCS for:

- **Raw data storage**: Centralized data sources
- **Intermediate datasets**: Pipeline outputs between stages
- **Artifact storage**: MLflow artifacts and model files
- **Cache storage**: Distributed caching for embeddings

### BigQuery Integration

BigQuery is used for:

- **Structured data storage**: Tabular datasets with schema validation
- **Analytics queries**: Complex data analysis and reporting
- **Data sharing**: Cross-team data access and collaboration
- **Cost optimization**: Efficient storage and query patterns

### Example BigQuery Dataset Configuration

```yaml
_bigquery_ds: &_bigquery_ds
  type: matrix.datasets.gcp.SparkDatasetWithBQExternalTable
  project_id: ${oc.env:RUNTIME_GCP_PROJECT_ID}
  dataset: release_${globals:versions.release}
  file_format: parquet
  save_args:
    mode: overwrite
    labels:
      git_sha: ${globals:git_sha}
```

## Environment Variables

The cloud environment requires specific environment variables:

### Required Variables

```bash
# GCP Configuration
RUNTIME_GCP_PROJECT_ID=your-project-id
RUNTIME_GCP_BUCKET=your-bucket-name

# Pipeline Configuration
RUN_NAME=your-run-name
RELEASE_VERSION=v1.0.0
RELEASE_FOLDER_NAME=releases

# MLflow Configuration
MLFLOW_EXPERIMENT_ID=1
WORKFLOW_ID=your-workflow-id
```

### Optional Variables

```bash
# Advanced Configuration
GIT_SHA=commit-hash
MLFLOW_ENDPOINT=your-mlflow-endpoint
```

## Data Flow Architecture

### Pipeline Stages in Cloud

1. **Ingestion**: Reads from GCS raw data buckets
2. **Integration**: Processes and stores in BigQuery
3. **Filtering**: Applies business logic filters
4. **Embeddings**: Generates and caches embeddings in GCS
5. **Modelling**: Trains models with cloud resources
6. **Evaluation**: Assesses model performance
7. **Inference**: Generates predictions
8. **Data Release**: Publishes results to BigQuery

### Data Path Structure

```yaml
paths:
  # Raw data (read-only from central buckets)
  raw: ${dev_gcs_bucket}/kedro/data/01_raw
  kg_raw: ${dev_gcs_bucket}/data/01_RAW
  
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

This ensures that:
- **Run names are consistent** across distributed workers
- **Artifacts are stored** in cloud storage
- **Experiments are shared** across team members
- **Tracking is resilient** to container restarts

## Kubernetes Integration

### Argo Workflows

The cloud environment is designed to run on Argo Workflows:

- **Distributed execution**: Multiple nodes run in parallel
- **Resource management**: Efficient use of cloud resources
- **Fault tolerance**: Automatic retries and error handling
- **Monitoring**: Integrated with GCP monitoring

### Container Configuration

```dockerfile
# Example Dockerfile for cloud deployment
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy pipeline code
COPY . .

# Set environment for cloud execution
ENV KEDRO_ENV=cloud
```

## Performance Optimizations

### Data Processing

- **Spark on Kubernetes**: Distributed data processing
- **BigQuery external tables**: Efficient data access patterns
- **GCS caching**: Reduced data transfer costs
- **Parallel execution**: Multi-stage pipeline optimization

### Resource Management

- **Auto-scaling**: Dynamic resource allocation
- **Cost optimization**: Efficient storage and compute usage
- **Monitoring**: Real-time performance tracking
- **Alerting**: Proactive issue detection

## Security Considerations

### Authentication

- **Service accounts**: GCP service account authentication
- **IAM roles**: Principle of least privilege
- **Secret management**: Secure credential storage
- **Network security**: VPC and firewall configuration

### Data Protection

- **Encryption**: Data encryption at rest and in transit
- **Access controls**: Fine-grained data access permissions
- **Audit logging**: Comprehensive access logging
- **Compliance**: HIPAA and other regulatory compliance

## Troubleshooting

### Common Cloud Issues

1. **Authentication errors**: Verify service account permissions
2. **Storage quota exceeded**: Check GCS bucket quotas
3. **BigQuery job failures**: Monitor query complexity and costs
4. **Network connectivity**: Verify VPC and firewall settings

### Debugging Commands

```bash
# Check GCP authentication
gcloud auth list

# Verify bucket access
gsutil ls gs://${RUNTIME_GCP_BUCKET}

# Test BigQuery access
bq ls --project_id=${RUNTIME_GCP_PROJECT_ID}

# Check MLflow connectivity
kedro mlflow ui --env cloud
```

### Monitoring and Logging

- **Cloud Logging**: Centralized log aggregation
- **Cloud Monitoring**: Performance metrics and alerts
- **Error reporting**: Automated error tracking
- **Cost monitoring**: Resource usage optimization

## Best Practices

1. **Use environment variables**: Never hardcode credentials
2. **Monitor costs**: Track resource usage and optimize
3. **Implement proper error handling**: Graceful failure recovery
4. **Use appropriate resource limits**: Prevent runaway costs
5. **Regular backups**: Ensure data durability
6. **Security reviews**: Regular access control audits

The cloud environment enables production-scale MATRIX pipeline execution with enterprise-grade reliability, scalability, and security features.
