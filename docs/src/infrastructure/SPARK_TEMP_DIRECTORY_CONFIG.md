# Spark Temporary Directory Configuration

This document explains the configuration changes made to ensure all Spark temporary operations use the mounted scratch volume `/data` instead of the container's ephemeral storage.

## Problem
Kubernetes pods were running out of ephemeral storage because Spark was writing temporary files to the container's filesystem instead of the mounted scratch volume. This caused pod evictions with errors like:
```
The node was low on resource: ephemeral-storage. Container main was using 167124244Ki
```

The issue occurred because Spark wasn't utilizing the available scratch storage mounted at `/data`, instead defaulting to writing temporary files to the container's limited ephemeral storage. This became critical during large data processing operations where Spark needed substantial temporary space for operations like shuffling, spilling, and intermediate data storage and also the fact that most of the time the workload was assigned to a Node Pool (VM) that already was used previously and the disk is not emptied.

## Solution
We implemented a comprehensive approach to force all temporary operations to use the mounted scratch volume `/data`:

### 1. Enhanced Spark Configuration (`conf/base/spark.yml`)
Modified Spark configurations to explicitly direct all temporary operations to the mounted volume:

```yaml
# Configure Spark to use mounted volume for temporary storage
spark.local.dir: /data/spark-temp

# Additional configurations to ensure all Spark temp operations use the mounted volume
spark.sql.warehouse.dir: /data/spark-warehouse
spark.sql.streaming.checkpointLocation: /data/checkpoints
spark.sql.streaming.stateStore.providerClass: org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider
spark.sql.streaming.stateStore.maintenanceInterval: 600s

# Force Spark to use the scratch volume for shuffle operations
spark.sql.adaptive.shuffle.localShuffleReader.enabled: false
spark.shuffle.service.enabled: false

# Ensure broadcast temp files use the mounted volume
spark.broadcast.compress: true
spark.io.compression.codec: lz4

# Additional temp directory configurations
spark.executor.logs.rolling.strategy: time
spark.executor.logs.rolling.time.interval: daily
spark.executor.logs.rolling.maxRetainedFiles: 1
# Critical: Force all serialization and spill operations to use our temp directory
spark.serializer.objectStreamReset: 100
spark.sql.execution.arrow.maxRecordsPerBatch: 5000
# Memory management to reduce spill to disk
spark.sql.adaptive.coalescePartitions.enabled: true
spark.sql.adaptive.coalescePartitions.minPartitionSize: 16MB
```

### 2. Runtime Spark Session Enhancement (`src/matrix/hooks.py`)
Added logic to dynamically configure additional temp directories when `SPARK_LOCAL_DIRS` is detected:

```python
# Ensure all temporary operations use the mounted volume when SPARK_LOCAL_DIRS is set
if os.environ.get("SPARK_LOCAL_DIRS") is not None:
    spark_local_dirs = os.environ["SPARK_LOCAL_DIRS"]
    logger.info(f"SPARK_LOCAL_DIRS detected: {spark_local_dirs}. Configuring additional temp paths.")
    
    # Override any relative temp paths to use the mounted volume
    temp_configs = {
        "spark.sql.warehouse.dir": f"{spark_local_dirs}/spark-warehouse",
        "spark.sql.streaming.checkpointLocation": f"{spark_local_dirs}/checkpoints",
        "java.io.tmpdir": spark_local_dirs,
        "spark.driver.host.tmpdir": spark_local_dirs,
        "spark.executor.host.tmpdir": spark_local_dirs,
    }
    
    for config_key, config_value in temp_configs.items():
        spark_conf.set(config_key, config_value)
        logger.info(f"Set {config_key} = {config_value}")
```

### 3. Argo Workflow Template Environment Variables (`templates/argo_wf_spec.tmpl`)
Enhanced the Argo workflow templates to set comprehensive system-level temp directories:

```yaml
# Environment variables to force all temporary operations to use `/data` (scratch volume)
- name: SPARK_LOCAL_DIRS
  value: /data/spark-temp
- name: SPARK_DRIVER_MEMORY
  value: {% raw %} "{{inputs.parameters.memory_limit}}"
  {% endraw %}
- name: JAVA_OPTS
  value: -Djava.io.tmpdir=/data/tmp
- name: _JAVA_OPTIONS
  value: -Djava.io.tmpdir=/data/tmp
- name: TMPDIR
  value: /data/tmp
- name: TMP
  value: /data/tmp
- name: TEMP
  value: /data/tmp
```

The workflow template also includes:
- Creation of necessary directories: `mkdir -p /data/tmp /data/spark-temp /data/spark-warehouse /data/checkpoints`
- Proper volume mounting of the scratch ephemeral volume to `/data`

## Files Modified

1. **`conf/base/spark.yml`** - Updated Spark configuration with explicit temp directory paths
2. **`src/matrix/hooks.py`** - Added runtime Spark session configuration for `SPARK_LOCAL_DIRS` detection
3. **`templates/argo_wf_spec.tmpl`** - Enhanced workflow template with comprehensive environment variables
4. **`tests/test_argo.py`** - Added tests to verify Spark and temp directory configuration

## Expected Behavior

When `SPARK_LOCAL_DIRS=/data/spark-temp` is set in the Kubernetes environment:

1. **Spark temp directories** will use `/data/spark-temp`, `/data/spark-warehouse`, `/data/checkpoints`, etc.
2. **Java temp operations** will use `/data/tmp` via `java.io.tmpdir` system properties
3. **System temp directories** will use `/data/tmp` via `TMPDIR`, `TMP`, `TEMP` environment variables
4. **All temporary files** should be written to the mounted scratch volume instead of ephemeral storage
5. **Container startup** creates necessary directories before Kedro execution

## Why These Changes Were Necessary

The primary motivation for these changes was that **Spark was not using the scratch storage** that was already available and mounted at `/data`. This caused several critical issues:

1. **Resource Exhaustion**: Pods were being evicted due to ephemeral storage limits being exceeded
2. **Performance Degradation**: Writing to container filesystem instead of optimized scratch volumes
3. **Reliability Issues**: Large Spark jobs failing due to insufficient temporary space

The comprehensive approach ensures that:
- Spark configuration explicitly directs temporary operations to scratch storage
- Runtime detection and override of temp paths when `SPARK_LOCAL_DIRS` is available
- System-level environment variables catch any remaining temporary operations
- All levels of the stack (Spark, JVM, OS) are configured consistently

## Verification

To verify the configuration is working:

1. **Check Spark session logs** for temp directory assignments showing `/data/spark-temp`
2. **Monitor pod ephemeral storage usage** (should remain minimal, under container limits)
3. **Verify scratch volume usage** increases during Spark operations at `/data`
4. **Look for environment variables** in pod specifications: `SPARK_LOCAL_DIRS=/data/spark-temp`
5. **Check directory creation** in container startup logs: `mkdir -p /data/tmp /data/spark-temp /data/spark-warehouse /data/checkpoints`

## Testing

The configuration is verified through automated tests in `test_argo.py`:
- `test_spark_and_temp_directory_environment_variables()` ensures all required environment variables are properly set
- Tests verify `SPARK_LOCAL_DIRS`, `JAVA_OPTS`, `TMPDIR`, `TMP`, and `TEMP` are configured correctly

This comprehensive approach ensures that Spark utilizes the available scratch storage effectively, preventing ephemeral storage exhaustion and improving overall pipeline reliability.
