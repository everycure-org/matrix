# --from-run Parameter Usage Examples

This document demonstrates how to use the new `--from-run` parameter to make the matrix pipeline more flexible.

## Overview

The `--from-run` parameter allows you to read input datasets from a specified run while writing outputs to the current run. This is particularly useful for debugging scenarios where you want to reproduce a pipeline run that failed in production.

## Basic Usage

```bash
# Read all input datasets from 'my-old-run' and write outputs to 'my-new-run'
kedro run -e test --from-run=my-old-run --run-name=my-new-run
```

## Examples

### 1. Debugging a Failed Pipeline Run

If a pipeline run failed in production and you want to debug it locally:

```bash
# Run the same pipeline locally, reading from the failed production run
kedro run -e test --from-run=failed-production-run-123 --nodes create_int_known_pairs
```

### 2. Testing Different Models with Same Input Data

```bash
# Run a new model configuration using the same input data from a previous run
kedro run -e test --from-run=baseline-run --nodes create_xg_synth_model
```

### 3. Cloud Environment Usage

```bash
# Read from a cloud run while writing to a new cloud run
kedro run -e cloud --from-run=production-run-v1.2 --run-name=debug-run-v1.3
```

## How It Works

The `--from-run` parameter:

1. **Dynamically discovers** which datasets in the catalog are run-based (use `${run_name}` in their path)
2. **Sets environment variables** to redirect all input datasets to the specified run
3. **Preserves output paths** to write to the current run

### Environment Variable Mapping

For example, when you run:
```bash
kedro run -e test --from-run=my-old-run
```

The system automatically sets these environment variables:
- `FILTERING=data/test/releases/test-release/runs/my-old-run/datasets/filtering`
- `EMBEDDINGS=data/test/releases/test-release/runs/my-old-run/datasets/embeddings`
- `MODELLING=data/test/releases/test-release/runs/my-old-run/datasets/modelling`
- `EVALUATION=data/test/releases/test-release/runs/my-old-run/datasets/evaluation`
- `MATRIX_GENERATION=data/test/releases/test-release/runs/my-old-run/datasets/matrix_generation`
- `MATRIX_TRANSFORMATIONS=data/test/releases/test-release/runs/my-old-run/datasets/matrix_transformations`
- `INFERENCE=data/test/releases/test-release/runs/my-old-run/datasets/inference`

## Configuration

The system automatically detects run-based datasets by examining the `globals.yml` configuration files. Any path that contains `${run_name}` is considered run-based and will be redirected when using `--from-run`.

### Test Environment (`conf/test/globals.yml`)
```yaml
paths:
  filtering:          ${oc.env:FILTERING, data/test/releases/test-release/runs/${run_name}/datasets/filtering}
  embeddings:         ${oc.env:EMBEDDINGS, data/test/releases/test-release/runs/${run_name}/datasets/embeddings}
  modelling:          ${oc.env:MODELLING, data/test/releases/test-release/runs/${run_name}/datasets/modelling}
  # ... other run-based paths
```

### Cloud Environment (`conf/cloud/globals.yml`)
```yaml
paths:
  filtering:          ${oc.env:FILTERING, ${run_dir}/datasets/filtering}
  embeddings:         ${oc.env:EMBEDDINGS, ${run_dir}/datasets/embeddings}
  modelling:          ${oc.env:MODELLING, ${run_dir}/datasets/modelling}
  # ... other run-based paths
```

## Benefits

1. **Flexibility**: No need to manually specify which datasets to read from which run
2. **Automatic Discovery**: The system automatically detects which datasets are run-based
3. **Environment Agnostic**: Works across all environments (test, cloud, sample, etc.)
4. **Backward Compatible**: Existing functionality remains unchanged
5. **Debugging Friendly**: Easy to reproduce production issues locally

## Migration from Old Parameters

The new `--from-run` parameter replaces the old specific parameters:
- ~~`--filtering-run`~~ → `--from-run` (affects all run-based datasets)
- ~~`--embeddings-run`~~ → `--from-run` (affects all run-based datasets)

## Error Handling

If the system cannot dynamically discover run-based paths (e.g., due to configuration issues), it falls back to a predefined list of known run-based paths to ensure functionality. 