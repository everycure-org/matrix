---
title: Test Environment
--- 

# Test Environment Guide

The test environment is specifically designed for fast, reliable testing of the MATRIX pipeline with synthetic data. It uses intentionally "broken" parameters that don't make sense for production but allow for rapid validation of pipeline logic and integration testing.

Test environment is used for **integration testing**, **CI/CD validation** and fast iteration whilst developing or debugging the pipeline. 

## Fabricated Data Strategy

The test environment uses fabricated data that:

- **Mimics real data structure** without sensitive information
- **Reduces processing time** significantly (minutes vs hours)
- **Maintains data relationships** for meaningful testing
- **Eliminates external dependencies** on cloud storage or APIs

As mentioned in the environments overview, the test environment uses parameters that "break" the meaning of algorithms:

```yaml
# Example: Reduced dimensionality for speed
embeddings.dimensionality_reduction:
  transformer:
    k: 2  # In Base Env we use 100

# Example: Minimal embedding dimensions
embeddings.topological:
  estimator:
    args:
      embeddingDimension: 3  # In Base Env we use 512

# Example: Reduced model tuning iterations
modelling.rf:
  model_options:
    model_tuning_args:
      tuner:
        n_calls: 10  # Base env: 100+
```
These are not supposed to make methodological sense but they are supposed to enable one fast testing of their code.

## Test-Specific Globals

Test environment has its specific `globals.yaml` file where all release environments and endpoints point to fabricated data. The data catalog directory mimics the base directory with exception of a `test` parent directory, allowing users to separate fabricated data products from real ones.
```yaml
# Test environment globals.yml
run_name: test-run
versions:
  release: test-release

# Local test data paths
paths:
  raw: data/test/raw
  kg_raw: data/test/raw
  ingestion: data/test/ingestion
  integration: data/test/releases/${versions.release}/datasets/integration
  filtering: data/test/releases/${versions.release}/runs/${run_name}/datasets/filtering
  embeddings: data/test/releases/${versions.release}/runs/${run_name}/datasets/embeddings
  modelling: data/test/releases/${versions.release}/runs/${run_name}/datasets/modelling
  evaluation: data/test/releases/${versions.release}/runs/${run_name}/datasets/evaluation
  matrix_generation: data/test/releases/${versions.release}/runs/${run_name}/datasets/matrix_generation
  inference: data/test/releases/${versions.release}/runs/${run_name}/datasets/inference
  tmp: data/test/tmp
  cache: data/test/cache
```

### Mock Services

The test environment includes mock configurations for external services such as API calls. This is because our pipeline heavily relies on OpenAI or node normalization services.

```yaml
# Mock OpenAI endpoint for testing
openai:
  endpoint: ${oc.env:OPENAI_ENDPOINT, http://localhost:1080/v1}
  api_key: dummy

# Dummy resolver for embeddings
embeddings.node:
  resolver:
    _object: matrix.pipelines.embeddings.encoders.DummyResolver
  api: "foo"
```

### Test Data Generation

The test environment relies on the fabricator pipeline:

```bash
# Generate test data
kedro run -p fabricator --env test
```
The fabricator data produces 'raw' data products which mimic real raw data products that are located in our GCS buckets. Therefore the data catalog entries for raw data differ significantly from base environment.

Instructions on how to use fabricator in detail can be found in [fabricator walkthrough](walkthroughs/fabricator.md)

