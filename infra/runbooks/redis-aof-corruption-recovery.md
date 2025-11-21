# Redis AOF Corruption Recovery Runbook

## Problem Description

Redis fails to start with AOF (Append-Only File) corruption errors:

```
# Bad file format reading the append only file appendonly.aof.X.incr.aof:
# make a backup of your AOF file, then use ./redis-check-aof --fix <filename.manifest>
```

This leads to:

- Redis pod in crash loop (continuous restarts)
- LiteLLM caching failures with `MISCONF` errors
- Redis refusing write operations: `MISCONF Redis is configured to save RDB snapshots, but it's currently unable to persist to disk`

## Root Causes

1. **Disk space exhaustion** during background save operations
2. **Corrupted AOF files** from unclean shutdowns or I/O errors
3. **Volume resizing** that preserves old corrupted data
4. **Insufficient storage** for Redis snapshots + AOF files

## Symptoms

- Redis logs show "Bad file format reading the append only file"
- Pod restarts repeatedly (check with `kubectl get pods -n redis`)
- LiteLLM errors: `redis_cache.py:440 - LiteLLM Redis Caching: async set() - Got exception from REDIS MISCONF`
- Redis refuses write commands when accessed via `redis-cli`

## Diagnosis Steps

### 1. Check Redis Pod Status

```bash
kubectl get pods -n redis
kubectl logs -n redis redis-0 --tail=50
```

Look for:

- Restart count > 0
- "Bad file format" errors
- "MISCONF" errors

### 2. Check Disk Usage

```bash
kubectl exec -n redis redis-0 -- df -h /data
kubectl exec -n redis redis-0 -- ls -lah /data
```

Look for:

- Disk usage > 70%
- Large `dump.rdb` or `appendonly.aof.*` files
- Multiple AOF files accumulating

### 3. Check Redis Configuration

```bash
kubectl exec -n redis redis-0 -- redis-cli CONFIG GET save appendonly appendfsync stop-writes-on-bgsave-error
```

## Recovery Procedures

### Option 1: Fix Corrupted AOF (Data Preservation)

If Redis pod is running but unstable:

```bash
# Access the pod
kubectl exec -it -n redis redis-0 -- sh

# Navigate to data directory
cd /data

# Fix the corrupted AOF file (replace X with actual number)
redis-check-aof --fix appendonly.aof.X.incr.aof

# Restart Redis
exit
kubectl delete pod redis-0 -n redis
```

### Option 2: Clean Slate Recovery (Recommended for Cache Usage)

When Redis is in crash loop and data is not critical (cache can be regenerated):

```bash
# 1. Scale down Redis
kubectl scale statefulset redis -n redis --replicas=0

# 2. Wait for pod termination
kubectl wait --for=delete pod/redis-0 -n redis --timeout=60s

# 3. Create cleanup pod to remove corrupted files
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: redis-pvc-cleanup
  namespace: redis
spec:
  containers:
  - name: cleanup
    image: busybox
    command: ['sh', '-c', 'rm -f /data/appendonly.aof.* /data/appendonlydir/* && ls -lah /data/ && sleep 3600']
    volumeMounts:
    - name: redis-data
      mountPath: /data
  volumes:
  - name: redis-data
    persistentVolumeClaim:
      claimName: redis-redis-0
  restartPolicy: Never
EOF

# 4. Verify cleanup
sleep 5
kubectl logs -n redis redis-pvc-cleanup

# 5. Delete cleanup pod and scale Redis back up
kubectl delete pod redis-pvc-cleanup -n redis
kubectl scale statefulset redis -n redis --replicas=1

# 6. Verify Redis is healthy
kubectl get pods -n redis
kubectl logs -n redis redis-0 --tail=20
```

Look for: "Ready to accept connections"

### Option 3: Complete Data Wipe

If corruption persists or you want a completely fresh start:

```bash
# 1. Delete the StatefulSet and PVC
kubectl delete statefulset redis -n redis
kubectl delete pvc redis-redis-0 -n redis

# 2. Force ArgoCD to recreate resources
kubectl delete application redis -n argocd
# Wait 30 seconds
kubectl apply -f infra/argo/app-of-apps/templates/redis.yaml

# 3. Wait for ArgoCD to sync
kubectl get application redis -n argocd -w
```

## Prevention Measures

### 1. Proper Storage Sizing

Ensure adequate storage (current: 10Gi):

```yaml
# infra/argo/app-of-apps/templates/redis.yaml
storageSpec:
  volumeClaimTemplate:
    spec:
      resources:
        requests:
          storage: 10Gi # Adjust based on cache size needs
```

### 2. Disable Write Blocking on Save Errors

For cache workloads where availability > durability:

```yaml
externalConfig:
  enabled: true
  data: |
    stop-writes-on-bgsave-error no
```

### 3. Configure Appropriate Persistence

Balance between durability and performance:

```yaml
externalConfig:
  enabled: true
  data: |
    # RDB snapshots - less frequent for cache
    save 900 1
    save 300 10
    save 60 10000
    # AOF for better durability
    appendonly yes
    appendfsync everysec
    # Don't block on save failures
    stop-writes-on-bgsave-error no
```

### 4. Monitor Disk Usage

Set up alerts for:

- Redis disk usage > 70%
- Redis pod restart count > 0
- AOF file size growth rate

Example Prometheus alert:

```yaml
- alert: RedisHighDiskUsage
  expr: kubelet_volume_stats_used_bytes{persistentvolumeclaim="redis-redis-0"} / kubelet_volume_stats_capacity_bytes{persistentvolumeclaim="redis-redis-0"} > 0.7
  for: 5m
  annotations:
    summary: "Redis disk usage above 70%"
```

## Configuration Reference

### Current Production Configuration

Location: `infra/argo/app-of-apps/templates/redis.yaml`

```yaml
storageSpec:
  volumeClaimTemplate:
    spec:
      resources:
        requests:
          storage: 10Gi
externalConfig:
  enabled: true
  data: |
    # Enable RDB snapshots with reasonable frequency
    save 900 1
    save 300 10
    save 60 10000
    # Enable AOF for better durability
    appendonly yes
    appendfsync everysec
    # Allow Redis to handle background save errors gracefully
    stop-writes-on-bgsave-error no
```

### Alternative: Ephemeral Cache (No Persistence)

For pure caching where data loss is acceptable:

```yaml
externalConfig:
  enabled: true
  data: |
    # Disable all persistence
    save ""
    appendonly no
    stop-writes-on-bgsave-error no
```

## Verification Steps

After recovery:

```bash
# 1. Check pod is running without restarts
kubectl get pods -n redis
# Expected: redis-0 should be Running with 0 restarts

# 2. Test Redis connectivity
kubectl exec -n redis redis-0 -- redis-cli PING
# Expected: PONG

# 3. Verify write operations work
kubectl exec -n redis redis-0 -- redis-cli SET test-key "hello"
kubectl exec -n redis redis-0 -- redis-cli GET test-key
# Expected: "hello"

# 4. Check LiteLLM can write to cache
# Run a LiteLLM request and check logs for cache writes

# 5. Monitor for 10 minutes
kubectl logs -n redis redis-0 --tail=50 -f
# Should not see AOF errors or save failures
```

## Related Issues

- **AIP-654**: Redis AOF corruption causing LiteLLM cache failures
- **Volume resize from 1GB → 100GB**: Preserved corrupted AOF files
- **Disk exhaustion**: 70% usage with 326MB RDB file on 973MB volume

## References

- [Redis Persistence Documentation](https://redis.io/docs/management/persistence/)
- [Redis AOF](https://redis.io/docs/management/persistence/#append-only-file)
- [LiteLLM Redis Configuration](../infrastructure/LiteLLM-Admin-Guide.md#5-redis-integration)
- [Redis Setup](../infrastructure/Redis-Setup.md)
