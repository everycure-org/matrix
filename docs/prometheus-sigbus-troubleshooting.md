# Prometheus Troubleshooting Guide

## SIGBUS Error Troubleshooting

### Symptom
Prometheus pods are in `CrashLoopBackOff` state with SIGBUS errors in the logs.

### Root Cause Analysis

**SIGBUS (Signal Bus Error)** in Prometheus is commonly caused by:

1. **Disk Full (Most Common)** - When the persistent volume is at 100% capacity
2. Memory mapping issues
3. Hardware/storage subsystem problems
4. Corrupted data files

### Diagnostic Steps

#### 1. Check Disk Usage First
Before investigating other causes, always check if the Prometheus persistent volume is full:

```bash
# Method 1: Create a debug pod to check disk usage
kubectl run disk-check --image=alpine --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"disk-check","image":"alpine","command":["df","-h","/data"],"volumeMounts":[{"name":"prometheus-data","mountPath":"/data"}]}],"volumes":[{"name":"prometheus-data","persistentVolumeClaim":{"claimName":"prometheus-kube-prometheus-stack-prometheus-db-prometheus-kube-prometheus-stack-prometheus-0"}}]}}' \
  -n observability

# Check the output
kubectl logs disk-check -n observability

# Clean up
kubectl delete pod disk-check -n observability

# Method 2: If Prometheus pod is running (but crashing), exec into it
kubectl exec prometheus-kube-prometheus-stack-prometheus-0 -n observability -c prometheus -- df -h /prometheus
```

#### 2. Check PVC Status
```bash
# Check PVC details
kubectl get pvc -n observability | grep prometheus
kubectl describe pvc prometheus-kube-prometheus-stack-prometheus-db-prometheus-kube-prometheus-stack-prometheus-0 -n observability
```

#### 3. Check Pod Logs
```bash
# Check Prometheus container logs for SIGBUS errors
kubectl logs prometheus-kube-prometheus-stack-prometheus-0 -n observability -c prometheus --previous
```

### Resolution

#### If Disk is Full (100% usage):

**Option 1: Expand the PVC (Recommended)**
```bash
# Patch the PVC to increase size (requires storage class with allowVolumeExpansion: true)
kubectl patch pvc prometheus-kube-prometheus-stack-prometheus-db-prometheus-kube-prometheus-stack-prometheus-0 \
  -n observability \
  --type='merge' \
  -p='{"spec":{"resources":{"requests":{"storage":"500Gi"}}}}'

# Wait for expansion to complete
kubectl get pvc -n observability -w

# Restart Prometheus pod to pick up the expanded volume
kubectl delete pod prometheus-kube-prometheus-stack-prometheus-0 -n observability
```

**Option 2: Clean up old data**
```bash
# If expansion is not possible, you may need to clean up old metrics data
# WARNING: This will result in data loss
kubectl exec prometheus-kube-prometheus-stack-prometheus-0 -n observability -c prometheus -- \
  find /prometheus -name "*.db" -mtime +30 -delete
```

**Option 3: Data Migration**
If PVC expansion is not supported by your storage class, you may need to migrate to a new, larger PVC.

#### If Disk is Not Full:
1. Check for memory limits and increase if necessary
2. Examine storage class and underlying storage health
3. Check for corrupted Prometheus data files
4. Consider downgrading Prometheus version if the issue started after an upgrade

### Prevention

#### 1. Monitor Disk Usage
Set up alerts for Prometheus disk usage:

```yaml
# Prometheus alerting rule
- alert: PrometheusVolumeAlmostFull
  expr: (prometheus_tsdb_symbol_table_size_bytes + prometheus_tsdb_head_series) / prometheus_tsdb_wal_size_bytes > 0.8
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Prometheus volume is almost full"
    description: "Prometheus volume usage is above 80%"
```

#### 2. Configure Retention Policies
Ensure appropriate data retention settings in your Prometheus configuration:

```yaml
# In prometheus.yml or Prometheus CRD
retention: "30d"  # Adjust based on your needs
retentionSize: "450GB"  # Set to ~90% of volume size
```

#### 3. Enable Automatic PVC Expansion
Consider implementing an automatic PVC expansion system that monitors disk usage and expands volumes before they reach 100%.

### Notes
- **SIGBUS errors due to disk full are immediately resolved** once disk space is available
- Always check disk space first as it's the most common and easily fixable cause
- PVC expansion is non-disruptive and preferred over data migration
- Ensure your storage class supports `allowVolumeExpansion: true` for easy expansion

### Example Case
In our investigation, we found:
- Prometheus PVC was 100% full (99GB out of 99GB used)
- This caused SIGBUS errors when Prometheus tried to write new data
- Expanding the PVC resolved the issue immediately

### Related Commands
```bash
# Check storage classes and their expansion capabilities
kubectl get storageclass

# Monitor PVC expansion progress
kubectl describe pvc <pvc-name> -n <namespace>

# Check Prometheus metrics about its own storage
kubectl port-forward prometheus-kube-prometheus-stack-prometheus-0 9090:9090 -n observability
# Then visit http://localhost:9090 and query: prometheus_tsdb_*
```
