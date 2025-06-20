---
title: Test Environment
--- 

# Test Environment Guide

The test environment is specifically designed for fast, reliable testing of the MATRIX pipeline with synthetic data. It uses intentionally "broken" parameters that don't make sense for production but allow for rapid validation of pipeline logic and integration testing.

## Purpose

The test environment serves several critical functions:

- **Integration Testing**: Validates that all pipeline components work together correctly
- **Fast Iteration**: Enables quick development cycles with minimal resource usage
- **CI/CD Validation**: Provides reliable automated testing for continuous integration
- **Pipeline Logic Verification**: Ensures programming correctness without production data complexity
- **Development Debugging**: Offers a controlled environment for troubleshooting

## Key Characteristics

### Synthetic Data Strategy

The test environment uses fabricated data that:

- **Mimics real data structure** without sensitive information
- **Reduces processing time** significantly (minutes vs hours)
- **Maintains data relationships** for meaningful testing
- **Eliminates external dependencies** on cloud storage or APIs

### "Broken" Parameters

As mentioned in the environments overview, the test environment uses parameters that "break" the meaning of algorithms:

```yaml
# Example: Reduced dimensionality for speed
embeddings.dimensionality_reduction:
  transformer:
    k: 2  # Production might use 100+

# Example: Minimal embedding dimensions
embeddings.topological:
  estimator:
    args:
      batchSize: 20      # Production: 1000+
      embeddingDimension: 3  # Production: 100+

# Example: Reduced model tuning iterations
modelling.rf:
  model_options:
    model_tuning_args:
      tuner:
        n_calls: 10  # Production: 100+
```

## Configuration Structure

### Test-Specific Globals

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

The test environment includes mock configurations for external services:

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

## Data Catalog Configuration

### Test Data Sources

The test environment uses simplified data sources:

```yaml
# Minimal drug and disease lists
ingestion.raw.disease_list:
  type: pandas.CSVDataset
  filepath: ${globals:paths.raw}/disease_list/translator/${globals:data_sources.disease_list.version}/source_disease_list.tsv

ingestion.raw.drug_list:
  type: pandas.CSVDataset
  filepath: ${globals:paths.raw}/drug_list/translator/${globals:data_sources.drug_list.version}/source_drug_list.tsv

# Ground truth data for validation
ingestion.raw.ground_truth.positives:
  type: pandas.CSVDataset
  filepath: ${globals:paths.kg_raw}/ground_truth/kg2/${globals:data_sources.gt.version}/tp_pairs.txt

ingestion.raw.ground_truth.negatives:
  type: pandas.CSVDataset
  filepath: ${globals:paths.kg_raw}/ground_truth/kg2/${globals:data_sources.gt.version}/tn_pairs.txt
```

## Pipeline Parameter Overrides

### Embeddings Pipeline

```yaml
# Reduced dimensionality for speed
embeddings.dimensionality_reduction:
  transformer:
    k: 2

# Minimal topological embeddings
embeddings.topological:
  estimator:
    args:
      batchSize: 20
      embeddingDimension: 3

# Reduced iterations
embeddings.topological_estimator:
  iterations: 1
  embedding_dim: 3
  walk_length: 2
```

### Modelling Pipeline

```yaml
# Reduced model tuning
modelling.rf:
  model_options:
    model_tuning_args:
      tuner:
        n_calls: 10

# Minimal unknown sample generation
modelling.xg_baseline:
  model_options:
    generator:
      n_unknown: 100
```

## Usage Patterns

### Running Tests

```bash
# Run full integration test
kedro run --env test -p test

# Run specific pipeline with test environment
kedro run --env test -p ingestion
kedro run --env test -p embeddings
kedro run --env test -p modelling

# Run with Makefile (recommended)
make integration_test
```

### Development Workflow

1. **Write code** in your development environment
2. **Run test environment** to validate changes
3. **Fix issues** identified by test environment
4. **Iterate** until tests pass
5. **Deploy** to production environments

### CI/CD Integration

The test environment is ideal for continuous integration:

```yaml
# Example GitHub Actions workflow
- name: Run Integration Tests
  run: |
    make integration_test
  env:
    KEDRO_ENV: test
```

## Performance Characteristics

### Execution Time

| Pipeline Stage | Test Environment | Production Environment |
|----------------|------------------|----------------------|
| **Ingestion** | ~30 seconds | ~30 minutes |
| **Integration** | ~1 minute | ~2 hours |
| **Embeddings** | ~2 minutes | ~6 hours |
| **Modelling** | ~1 minute | ~4 hours |
| **Total** | ~5 minutes | ~12+ hours |

### Resource Usage

- **Memory**: Minimal (2-4 GB vs 32+ GB)
- **CPU**: Single core sufficient vs multi-core cluster
- **Storage**: Local filesystem only
- **Network**: No external dependencies

## Validation Strategy

### What Gets Tested

1. **Pipeline Logic**: All nodes execute without errors
2. **Data Flow**: Correct data transformations between stages
3. **Integration**: Components work together properly
4. **Configuration**: Environment-specific settings are applied
5. **Error Handling**: Graceful failure modes

### What Doesn't Get Tested

1. **Production Performance**: Real-world scalability
2. **Data Quality**: Accuracy with real data
3. **External Services**: Cloud storage, APIs, etc.
4. **Resource Constraints**: Memory/CPU limitations

## Troubleshooting

### Common Test Issues

1. **Missing test data**: Ensure fabricator pipeline has run
2. **Parameter conflicts**: Check for environment-specific overrides
3. **Mock service failures**: Verify local service availability
4. **Path issues**: Confirm test data directory structure

### Debugging Commands

```bash
# Check test environment configuration
kedro info --env test

# Validate test data exists
ls -la data/test/

# Run specific node for debugging
kedro run --env test --node=ingestion.raw.disease_list

# Check pipeline structure
kedro pipeline list --env test
```

### Test Data Generation

The test environment relies on the fabricator pipeline:

```bash
# Generate test data
kedro run -p fabricator --env test

# Verify test data
kedro catalog list --env test
```

## Best Practices

### Development

1. **Always test locally** before pushing changes
2. **Use test environment** for initial validation
3. **Keep test parameters** intentionally small
4. **Document test data** requirements clearly

### Maintenance

1. **Update test data** when schema changes
2. **Monitor test execution time** for regressions
3. **Review parameter overrides** regularly
4. **Ensure test coverage** of new features

### Integration

1. **Run tests in CI/CD** for every change
2. **Fail fast** on test environment issues
3. **Use consistent test data** across environments
4. **Monitor test reliability** over time

## Comparison with Other Environments

| Aspect | Test Environment | Base Environment | Cloud Environment |
|--------|------------------|------------------|-------------------|
| **Purpose** | Fast testing | Local development | Production |
| **Data** | Synthetic | Real (local) | Real (cloud) |
| **Performance** | Optimized for speed | Balanced | Optimized for scale |
| **Resources** | Minimal | Moderate | High |
| **Dependencies** | None | Local services | Cloud services |
| **Execution Time** | Minutes | Hours | Hours to days |

The test environment is essential for maintaining code quality and enabling rapid development cycles in the MATRIX pipeline.
