# Spark Temporary Directory Configuration

This document explains the configuration changes made to ensure all Spark temporary operations use the mounted volume `/data` instead of the container's ephemeral storage.

## Problem
Kubernetes pods were running out of ephemeral storage because Spark was writing temporary files to the container's filesystem instead of the mounted scratch volume, causing pod evictions with errors like:
```
The node was low on resource: ephemeral-storage. Container main was using 167124244Ki
```

## Solution
We implemented a comprehensive approach to force all temporary operations to use the mounted volume `/data`:

### 1. Enhanced Spark Configuration (`conf/base/spark.yml`)
Added extensive Spark configurations to direct all temporary operations to the mounted volume:

```yaml
# Configure Spark to use mounted volume for temporary storage
spark.local.dir: ${oc.env:SPARK_LOCAL_DIRS,/tmp}

# Additional configurations to ensure all Spark temp operations use the mounted volume
spark.sql.warehouse.dir: ${oc.env:SPARK_LOCAL_DIRS,/tmp}/spark-warehouse
spark.sql.streaming.checkpointLocation: ${oc.env:SPARK_LOCAL_DIRS,/tmp}/checkpoints
spark.sql.streaming.stateStore.providerClass: org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider
spark.sql.streaming.stateStore.maintenanceInterval: 600s

# Force Spark to use the scratch volume for shuffle operations
spark.sql.adaptive.shuffle.localShuffleReader.enabled: false
spark.shuffle.service.enabled: false

# Ensure broadcast temp files use the mounted volume
spark.broadcast.compress: true
spark.io.compression.codec: lz4

# Configure spill directories to use mounted volume
spark.executor.logs.rolling.strategy: time
spark.executor.logs.rolling.time.interval: daily
spark.executor.logs.rolling.maxRetainedFiles: 1
```

### 2. Runtime Spark Session Enhancement (`src/matrix/hooks.py`)
Added logic to dynamically configure additional temp directories when `SPARK_LOCAL_DIRS` is detected:

```python
# Ensure all temporary operations use the mounted volume when SPARK_LOCAL_DIRS is set
if os.environ.get("SPARK_LOCAL_DIRS") is not None:
    spark_local_dirs = os.environ["SPARK_LOCAL_DIRS"]
    
    # Override any relative temp paths to use the mounted volume
    temp_configs = {
        "spark.sql.warehouse.dir": f"{spark_local_dirs}/spark-warehouse",
        "spark.sql.streaming.checkpointLocation": f"{spark_local_dirs}/checkpoints",
        "java.io.tmpdir": spark_local_dirs,
        "spark.driver.host.tmpdir": spark_local_dirs,
        "spark.executor.host.tmpdir": spark_local_dirs,
    }
```

### 3. System-Level Environment Variables
Enhanced the Argo workflow templates to set system-level temp directories:

```yaml
# Additional environment variables to force all temporary operations to use /data
- name: JAVA_OPTS
  value: -Djava.io.tmpdir=/data
- name: TMPDIR
  value: /data
- name: TMP
  value: /data
- name: TEMP
  value: /data
```

## Files Modified

1. **`conf/base/spark.yml`** - Enhanced Spark configuration
2. **`src/matrix/hooks.py`** - Runtime Spark session configuration
3. **`templates/argo-workflow-template.yml`** - Workflow environment variables
4. **`templates/argo_wf_spec.tmpl`** - Template environment variables

## Expected Behavior

When `SPARK_LOCAL_DIRS=/data` is set in the Kubernetes environment:

1. **Spark temp directories** will use `/data/spark-warehouse`, `/data/checkpoints`, etc.
2. **Java temp operations** will use `/data` via `java.io.tmpdir`
3. **System temp directories** will use `/data` via `TMPDIR`, `TMP`, `TEMP`
4. **All temporary files** should be written to the mounted scratch volume instead of ephemeral storage

## Verification

To verify the configuration is working:

1. Check Spark session logs for temp directory assignments
2. Monitor pod ephemeral storage usage (should be minimal)
3. Verify scratch volume usage increases during Spark operations
4. Look for the warning: `"spark.local.dir will be overridden by the value set by the cluster manager"`

This warning is **expected and correct** - it confirms that Kubernetes is properly overriding the Spark configuration with `SPARK_LOCAL_DIRS=/data`.
