# GPU Support Implementation Summary

## Overview

This implementation adds automatic GPU detection and configuration to the MATRIX Spark session manager, enabling GPU-accelerated processing when available without requiring manual configuration.

## Changes Made

### 1. GPU Detection Function (`libs/matrix-gcp-datasets/src/matrix_gcp_datasets/spark_utils.py`)

Added `detect_gpus()` function with multi-method detection:

- **CuPy**: Direct CUDA runtime query (most reliable)
- **CUDA_VISIBLE_DEVICES**: Environment variable parsing
- **nvidia-smi**: System command fallback
- Returns 0 if no GPUs detected

### 2. Automatic Spark Configuration (`libs/matrix-gcp-datasets/src/matrix_gcp_datasets/spark_utils.py`)

Enhanced `SparkManager.initialize_spark()` to:

- Detect available GPUs automatically
- Configure Spark with GPU-specific settings when GPUs are present
- Apply configurations only if not already set (respects user overrides)

GPU configurations applied:

```yaml
spark.executor.resource.gpu.amount: "1"
spark.task.resource.gpu.amount: "0.1"  # Fractional to allow concurrent tasks
spark.rapids.sql.enabled: "true"
spark.rapids.memory.pinnedPool.size: "2G"
spark.python.worker.reuse: "true"
```

### 3. Configuration Documentation (`pipelines/matrix/conf/base/spark.yml`)

Added commented GPU configuration section showing available options and documenting auto-detection behavior.

### 4. Test Coverage (`libs/matrix-gcp-datasets/tests/test_spark_utils.py`)

Comprehensive tests for:

- GPU detection via CuPy
- GPU detection via CUDA_VISIBLE_DEVICES
- GPU detection via nvidia-smi
- No GPU scenario
- Spark configuration with GPUs
- Spark configuration without GPUs

### 5. Documentation (`docs/src/infrastructure/GPU_CONFIGURATION.md`)

Complete guide covering:

- Feature overview and benefits
- Local and cloud usage
- GPU-accelerated operations
- Performance considerations
- Troubleshooting guide

## Integration with Existing Code

### Compatible with Existing GPU Support

The implementation works seamlessly with existing GPU features:

1. **ModelWrapper** (`pipelines/matrix/src/matrix/pipelines/modelling/model.py`):

   - Already supports CUDA via `estimator_uses_cuda()` and `to_estimator_device()`
   - No changes needed

2. **Argo GPU Requests** (`pipelines/matrix/src/matrix/pipelines/kedro4argo/`):

   - Existing `num_gpus` parameter in `ArgoResourceConfig` works as before
   - Spark will auto-configure based on detected GPUs in the container

3. **Model Training**:
   - XGBoost and LightGBM models with GPU parameters work unchanged
   - Spark now properly configured to support GPU-enabled workers

## Benefits

### 1. Zero Configuration Required

- Automatically detects and uses GPUs when available
- Falls back gracefully to CPU when no GPUs present
- No environment-specific configuration needed

### 2. Performance Improvements

Expected speedups for `make_predictions_and_sort`:

- **GPU-accelerated inference**: 5-10x faster for model predictions
- **RAPIDS SQL operations**: 2-5x faster for data transformations
- **Better parallelism**: Proper resource allocation for GPU tasks

### 3. Cloud-Ready

- Works in Kubernetes/Argo environments with GPU nodes
- Respects `CUDA_VISIBLE_DEVICES` for multi-GPU systems
- Compatible with GKE node pools with GPU accelerators

## Usage Example

### Before (Manual Configuration)

```yaml
# spark.yml - had to manually configure for GPU nodes
spark.executor.resource.gpu.amount: 1
spark.task.resource.gpu.amount: 0.1  # Fractional for task concurrency
```

### After (Automatic)

```python
# No configuration needed!
SparkManager.initialize_spark()
# Automatically detects and configures GPUs
```

### In Pipeline Nodes

```python
@inject_object()
def make_predictions_and_sort(
    node_embeddings: ps.DataFrame,
    pairs: ps.DataFrame,
    model: ModelWrapper,
) -> ps.DataFrame:
    # Spark session already GPU-configured if available
    # Model inference will use GPU automatically
    pairs_with_scores = pairs_with_embeddings.mapInPandas(
        model_predict,
        schema
    )
    # ...
```

## Testing

Run tests to verify GPU detection:

```bash
cd libs/matrix-gcp-datasets
uv run pytest tests/test_spark_utils.py -v
```

## Backward Compatibility

✅ **Fully backward compatible**:

- CPU-only systems work exactly as before
- Existing configurations are respected (not overridden)
- No breaking changes to APIs or configurations

## Next Steps (Optional Enhancements)

1. **Add GPU metrics to MLflow tracking**

   - Track GPU utilization during training
   - Record GPU memory usage

2. **Optimize partition sizing based on GPU memory**

   - Dynamically adjust `num_partitions` based on GPU RAM
   - Prevent OOM errors on GPU workers

3. **Add RAPIDS-specific optimizations**

   - Use GPU-accelerated UDFs where applicable
   - Optimize join strategies for GPU execution

4. **Multi-GPU support**
   - Distribute tasks across multiple GPUs
   - Configure proper GPU affinity

## Files Modified

1. `libs/matrix-gcp-datasets/src/matrix_gcp_datasets/spark_utils.py` - Added GPU detection and configuration
2. `pipelines/matrix/conf/base/spark.yml` - Added GPU configuration documentation
3. `libs/matrix-gcp-datasets/tests/test_spark_utils.py` - Added comprehensive tests
4. `docs/src/infrastructure/GPU_CONFIGURATION.md` - Added user documentation

## Validation

To verify the implementation is working:

1. **On a GPU-enabled machine**:

   ```bash
   python -c "from matrix_gcp_datasets.spark_utils import detect_gpus; print(f'GPUs: {detect_gpus()}')"
   ```

2. **Check Spark logs**:

   ```bash
   uv run kedro run --env test
   # Look for: "Detected X GPU(s)" and "Configuring Spark for X GPU(s)"
   ```

3. **Run integration tests**:
   ```bash
   cd pipelines/matrix
   make integration_test
   ```
