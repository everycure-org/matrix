# GPU Configuration for Spark

The MATRIX pipeline now includes automatic GPU detection and configuration for Apache Spark, enabling GPU-accelerated processing when available.

## Overview

The Spark session manager automatically detects available GPUs and configures Spark to use them for compatible operations. This can significantly speed up data processing, especially for ML inference tasks and large-scale data transformations.

## Features

### Automatic GPU Detection

The system detects GPUs through multiple methods (in order of precedence):

1. **CuPy**: Queries CUDA runtime directly if CuPy is installed
2. **CUDA_VISIBLE_DEVICES**: Reads the environment variable to determine available GPUs
3. **nvidia-smi**: Falls back to querying the NVIDIA System Management Interface

### Automatic Configuration

When GPUs are detected, Spark is automatically configured with:

```yaml
spark.executor.resource.gpu.amount: 1
spark.task.resource.gpu.amount: 0.1 # Fractional to enable task concurrency
spark.rapids.sql.enabled: true
spark.rapids.memory.pinnedPool.size: 2G
spark.python.worker.reuse: true
spark.sql.execution.arrow.pyspark.enabled: true
```

**Important**: `spark.task.resource.gpu.amount` is set to `0.1` (not `1.0`) to allow multiple tasks to share the same GPU concurrently. This prevents the GPU from becoming a bottleneck when you have many CPU cores. For example:

- With 31 cores and `spark.task.resource.gpu.amount: 1.0` → only 1 task runs at a time (wastes 30 cores)
- With 31 cores and `spark.task.resource.gpu.amount: 0.1` → up to 10 tasks can share the GPU in parallel

## Usage

### Local Development

No configuration needed! If you have a CUDA-capable GPU and the necessary drivers installed, the system will automatically detect and use it.

#### Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Driver** installed
3. **CUDA Toolkit** (optional but recommended)
4. **CuPy** (optional, for better detection):
   ```bash
   pip install cupy-cuda12x  # Replace with your CUDA version
   ```

### Cloud/Cluster Deployment

For Kubernetes/Argo deployments with GPU nodes:

1. **Request GPU resources** in your Argo workflow:

   ```python
   from matrix.pipelines.kedro4argo import ArgoNode, ArgoResourceConfig

   node = ArgoNode(
       func=my_node_function,
       inputs=["input_data"],
       outputs=["output_data"],
       name="gpu_enabled_node",
       argo_config=ArgoResourceConfig(
           num_gpus=1,  # Request 1 GPU
           memory_limit=32,
           cpu_limit=8
       )
   )
   ```

2. **Set CUDA_VISIBLE_DEVICES** (if needed) to control which GPUs are accessible:

   ```yaml
   env:
     - name: CUDA_VISIBLE_DEVICES
       value: "0" # Use only GPU 0
   ```

3. The Spark session will automatically configure itself based on detected GPUs.

## GPU-Accelerated Operations

### Model Inference

The `ModelWrapper` in `matrix.pipelines.modelling.model` already supports CUDA-accelerated inference:

```python
from matrix.pipelines.modelling.model import ModelWrapper

# Models with device="cuda" will automatically use GPU
model = ModelWrapper(
    estimators=[xgboost_model],  # XGBoost with tree_method="gpu_hist"
    agg_func=np.mean
)

# Predictions will run on GPU if available
predictions = model.predict_proba(X)
```

### PySpark Operations

With RAPIDS enabled, certain Spark SQL operations can be GPU-accelerated:

- Aggregations
- Joins
- Window functions
- UDFs (with appropriate setup)

## Monitoring

Check the Spark logs to verify GPU configuration:

```
INFO SparkManager: Detected 2 GPU(s) via CuPy
INFO SparkManager: Configuring Spark for 2 GPU(s)
INFO SparkManager: Setting GPU config spark.executor.resource.gpu.amount to 1
INFO SparkManager: Setting GPU config spark.task.resource.gpu.amount to 1
INFO SparkManager: Setting GPU config spark.rapids.sql.enabled to true
```

## Disabling GPU Usage

To disable automatic GPU detection and force CPU-only execution:

1. **Unset CUDA environment variables**:

   ```bash
   unset CUDA_VISIBLE_DEVICES
   ```

2. **Or explicitly disable in spark.yml**:
   ```yaml
   spark.executor.resource.gpu.amount: 0
   spark.rapids.sql.enabled: false
   ```

## Performance Considerations

### When GPU Acceleration Helps

- **Large batch inference**: Predicting on millions of drug-disease pairs
- **Matrix operations**: Large-scale embeddings and transformations
- **Complex aggregations**: Group-by operations on large datasets

### When GPU May Not Help

- **Small datasets**: Overhead of GPU data transfer may outweigh benefits
- **I/O-bound operations**: Reading/writing data from storage
- **Simple transformations**: Basic column operations

### Optimal Configuration

For the `make_predictions_and_sort` node:

```python
# Configure partitioning to match GPU count
num_gpus = detect_gpus()
num_partitions = max(num_gpus * 4, 32)  # 4x oversubscription

pairs_with_embeddings = pairs_with_embeddings.repartition(num_partitions)
```

## Troubleshooting

### GPUs Not Detected

1. **Check GPU availability**:

   ```bash
   nvidia-smi
   ```

2. **Verify CUDA installation**:

   ```bash
   nvcc --version
   ```

3. **Check Spark logs** for detection messages

### Out of Memory Errors

GPUs have limited memory. If you encounter OOM errors:

1. **Reduce partition size**:

   ```python
   pairs_with_embeddings = pairs_with_embeddings.repartition(num_partitions * 2)
   ```

2. **Reduce batch size** in model inference

3. **Adjust GPU memory pool**:
   ```yaml
   spark.rapids.memory.pinnedPool.size: 1G # Reduce from default 2G
   ```

### Performance Not Improving

1. **Check if models are using GPU**:

   - XGBoost: Set `tree_method="gpu_hist"`, `device="cuda"`
   - LightGBM: Set `device="gpu"`

2. **Verify RAPIDS is active**:

   ```python
   spark.conf.get("spark.rapids.sql.enabled")
   ```

3. **Profile your workload** to identify bottlenecks

## Related Documentation

- [Model Training and Inference](../pipeline/data_science/modelling.md)
- [Spark Configuration](./SPARK_TEMP_DIRECTORY_CONFIG.md)
- [Performance Optimization](../getting_started/deep_dive/performance.md)

## References

- [RAPIDS Accelerator for Apache Spark](https://nvidia.github.io/spark-rapids/)
- [CuPy Documentation](https://docs.cupy.dev/)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
