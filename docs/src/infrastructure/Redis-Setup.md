# Redis Setup & Connection Documentation

### 1. Redis Operator Deployment

- **Operator**: `redis-operator` Helm chart (ot-container-kit) version `0.22.0`
- **Namespace**: `redis` (auto-created via ArgoCD `CreateNamespace=true`)
- **Management**: Deployed and continuously reconciled by ArgoCD (sync-wave 7)

### 2. Redis Cluster Deployment

- **Redis Release**: `redis` Helm chart (ot-container-kit) version `0.16.5`
- **Namespace**: `redis` (ArgoCD application `redis`, sync-wave 9)
- **Mode**: (Assumed) Standalone or Sentinel-managed (verify with `kubectl get redis -n redis` if CRDs used, or `kubectl get sts -n redis` if statefulset based)
- **Service Discovery**: Core service DNS: `redis.redis.svc.cluster.local`
- **Port**: `6379`
- **Intended Uses**: Caching, rate limiting / coordination (e.g. used by `litellm` per application config)

### 3. Security & Access

- **TLS**: Not enabled by default unless chart configured for it (assumed disabled — enable in future hardening phase if required)
- **Network Access**: Cluster-internal only (ClusterIP service) — expose externally only via port-forward or jump tools.

### 4. ArgoCD Management

Both the operator and the Redis instance are deployed as ArgoCD `Application` resources and will automatically prune/repair drift.

To list them:

```bash
kubectl get applications.argoproj.io -n argocd | grep redis
```

---

## How to Connect to Redis

### Connection Details

```
Host: redis.redis.svc.cluster.local
Port: 6379
```

### From Within the Cluster (Recommended)

Most in-cluster workloads should use environment variables sourced from the secret:

```yaml
# Example deployment snippet
env:
  - name: REDIS_HOST
    value: "redis.redis.svc.cluster.local"
  - name: REDIS_PORT
    value: "6379"
```

### Port Forward for Local Development

```bash
# Forward local port 6379 to cluster Redis
kubectl port-forward -n redis svc/redis 6379:6379
# If the service name differs, list services: kubectl get svc -n redis
```

Then connect locally (CLI example below) using `localhost:6379`.

### Using redis-cli

```bash
# Install if needed (macOS)
brew install redis

# Test auth (replace with actual password if not exported)
redis-cli -h redis.redis.svc.cluster.local -p 6379 PING
# Expect: PONG
```

If port-forwarded:

```bash
redis-cli -h 127.0.0.1 -p 6379 INFO server | head
```

### Python Example (redis-py)

```python
import redis, os

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis.redis.svc.cluster.local"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    socket_timeout=3,
)

r.set("healthcheck", "ok", ex=60)
print("Value:", r.get("healthcheck").decode())
```

### Health & Monitoring

Basic checks:

```bash
kubectl get pods -n redis
kubectl get svc -n redis
kubectl logs -n redis -l app.kubernetes.io/name=redis --tail=100
```

If operator CRDs are present:

```bash
kubectl get redis -n redis || echo "(No Redis CRD objects found — using raw Helm chart resources)"
```

Memory / key metrics (requires redis-cli auth):

```bash
redis-cli INFO memory | grep used_memory_human
redis-cli DBSIZE
```

### Important Notes

1. Prefer in-cluster access over exposing Redis externally.
2. Store the password only in Kubernetes secrets; never commit credentials.
3. Consider enabling TLS + AUTH rotation in a future hardening phase.
4. For high availability / clustering needs, evaluate Sentinel or Redis Cluster mode (not yet enabled).
5. Rate-limiting / coordination components (e.g. `litellm`) rely on stable DNS: `redis.redis.svc.cluster.local`.

### Troubleshooting

#### Connection Refused

- Pod not ready: `kubectl get pods -n redis` and check readiness probes.
- Service name mismatch: `kubectl get svc -n redis`.
- Network Policy (if later introduced) blocking namespace.

#### High Latency / Timeouts

- Check pod resource limits; adjust values in Helm chart if under-provisioned.
- Run: `redis-cli INFO stats | grep instantaneous_ops_per_sec`

#### Persistence Issues

- Confirm PVCs: `kubectl get pvc -n redis`.
- Check StatefulSet (if used): `kubectl get sts -n redis`.

### Verifying Deployment via ArgoCD

```bash
kubectl get application -n argocd redis redis-operator
kubectl describe application -n argocd redis | grep -i sync
```

---

## Next Improvements (Future Work)

- Enable TLS and enforce in-transit encryption.
- Use Password protection.
- Add password rotation automation.
- Add Prometheus exporter (e.g. redis-exporter) and alerting rules.
- Evaluate HA deployment (Sentinel or Redis Cluster) if single instance becomes bottleneck.

---

## Quick Reference

| Item                | Value                                          |
| ------------------- | ---------------------------------------------- |
| Namespace           | `redis`                                        |
| Service (ClusterIP) | `redis` (DNS: `redis.redis.svc.cluster.local`) |
| Port                | 6379                                           |
| Secret (auth)       | `redis-auth` (assumed; verify)                 |
| Access Method       | In-cluster / kubectl port-forward              |

> Replace assumed names if actual Helm values override defaults.
