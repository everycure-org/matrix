# LiteLLM Setup & Connection Documentation

### 1. Deployment via ArgoCD

- **ArgoCD Application Name**: `litellm` (sync-wave 10)
- **Namespace**: `litellm` (auto-created with `CreateNamespace=true`)
- **Source Repo**: `https://github.com/BerriAI/litellm.git`
- **Chart Path**: `deploy/charts/litellm-helm`
- **Version Pin**: `targetRevision: v1.76.1-stable` (application), image tag `main-stable` for runtime container
- **Reconciliation**: Automated with prune + selfHeal

### 2. Runtime Configuration

- **Replicas**: 3 (`replicaCount: 3`)
- **Service Type**: ClusterIP (`port: 4000`)
- **Primary Dependencies**:
  - PostgreSQL (pooler RW service) for persistence / metadata: `postgresql-cloudnative-pg-cluster-pooler-rw.postgresql.svc.cluster.local`
  - Redis for routing state & caching: `redis.redis.svc.cluster.local:6379`
- **Secrets Referenced**:
  - `litellm-provider-keys` (LLM provider API keys like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
  - `postgres` (supplies `DATABASE_PASSWORD`)
  - `litellm-master-key` (master key secret; key name: `LITELLM_MASTER_KEY`)
- **Caching**: Redis enabled with TTL 86,400 seconds (1 day) & namespace `litellm_cache`
- **Retries**: `num_retries: 2` for routing
- **Telemetry**: Disabled (`telemetry: false`)
- **UI**: Enabled (`ui.enabled: true`) on same service port

### 3. Resource Management

- **Requests**: 250m CPU, 512Mi memory
- **Limits**: 1 CPU, 2Gi memory

### 4. Database Integration

`db.url` template (with secret expansion):

```
postgresql://litellm:$(DATABASE_PASSWORD)@postgresql-cloudnative-pg-cluster-pooler-rw.postgresql.svc.cluster.local:5432/app?schema=litellm
```

- **User**: `litellm`
- **Database**: `app`
- **Schema**: `litellm`
- **Credentials**: Injected via the `postgres` secret (`DATABASE_PASSWORD` key expected)

### 5. Redis Integration

Used both for:

- Rate limiting / router state (`router_settings`)
- Response / embedding cache (`litellm_settings.cache_params`)
  Configuration (no password currently set — recommend adding secret-backed auth):

```
Host: redis.redis.svc.cluster.local
Port: 6379
Cache TTL: 86400 seconds
Namespace: litellm_cache
Flush Size: 100
```

---

## Architecture Overview

```
(Client) --> [Service: litellm:4000] --> Model Proxy Logic
                                 |--> PostgreSQL (persistence: usage logs, metadata)
                                 |--> Redis (cache + routing state)
                                 |--> External LLM APIs (OpenAI, Anthropic, etc.)
```

High-level components:

1. **API / Proxy Layer**: Exposes OpenAI-compatible routes & management UI
2. **Credential Layer**: Provider keys pulled from Kubernetes secret(s)
3. **Caching Layer**: Redis for latency + quota coordination across replicas
4. **Persistence Layer**: PostgreSQL for structured storage
5. **Orchestration**: ArgoCD for drift correction & version pinning

---

## How to Access LiteLLM

### In-Cluster (Service DNS)

```
Host: litellm.litellm.svc.cluster.local
Port: 4000
Protocol: HTTP (considering TLS ingress in future)
```

Example curl (OpenAI-style chat completion):

```bash
curl -s \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  http://litellm.litellm.svc.cluster.local:4000/v1/chat/completions \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Say hello in 5 words"}
    ]
  }' | jq .
```

### Port Forward for Local Testing

```bash
kubectl port-forward -n litellm svc/litellm 4000:4000
# Now available at http://litellm.api.prod.everycure.org
```

Test locally:

```bash
curl -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
     -H "Content-Type: application/json" \
     http://litellm.api.prod.everycure.org/v1/models | jq .
```

### UI Access

Open in browser:

```
http://litellm.api.prod.everycure.org/
```

(Assess access control—if none, consider restricting with network policy or auth proxy.)

---

## Environment Variables & Secrets

Typical contents (verify actual secret keys):

| Secret                  | Purpose                  | Expected Keys                                           |
| ----------------------- | ------------------------ | ------------------------------------------------------- |
| `litellm-provider-keys` | External LLM providers   | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, others as needed |
| `postgres`              | Database credentials     | `DATABASE_PASSWORD`                                     |
| `litellm-master-key`    | Master API key for auth  | `LITELLM_MASTER_KEY`                                    |
| `redis-auth` (future)   | Redis password (not yet) | `password`                                              |

Retrieve master key:

```bash
kubectl get secret litellm-master-key -n litellm -o jsonpath='{.data.LITELLM_MASTER_KEY}' | base64 -d
```

Set locally:

```bash
export LITELLM_MASTER_KEY="$(kubectl get secret litellm-master-key -n litellm -o jsonpath='{.data.LITELLM_MASTER_KEY}' | base64 -d)"
```

Database password:

```bash
export DATABASE_PASSWORD="$(kubectl get secret postgres -n litellm -o jsonpath='{.data.DATABASE_PASSWORD}' | base64 -d)"
```

(Adjust namespace if `postgres` secret lives in `postgresql` namespace; if so, sync or project a copy into `litellm`.)

---

## Sample Client Usage (Python)

> When using outside GKE, `LITELLM_BASE` should be set to `http://litellm.litellm.svc.cluster.local:4000`

```python
import os, requests, json

base_url = os.getenv("LITELLM_BASE", "https://litellm.api.prod.everycure.org")
master_key = os.getenv("LITELLM_MASTER_KEY")

payload = {
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Return a JSON object with a greeting"}],
  "response_format": {"type": "json_object"}
}

resp = requests.post(
  f"{base_url}/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {master_key}",
    "Content-Type": "application/json"
  },
  data=json.dumps(payload),
  timeout=30,
)
print(resp.status_code)
print(resp.json())
```

---

## Scaling Considerations

- **Horizontal Scaling**: Increase `replicaCount`; Redis + Postgres must handle added concurrency.
- **Bottlenecks**: External provider API rate limits; enable per-provider throttling.
- **Cache Strategy**: TTL=1 day—validate memory pressure in Redis; adjust `ttl` or add LRU if needed.
- **Retries**: `num_retries: 2`—tune based on provider error patterns.

### Recommendations

1. Introduce HPA on CPU + custom metrics (req/sec) once baseline traffic known.
2. Add circuit breaking for provider timeouts.
3. Consider distinct Redis logical DB or namespace per environment.

---

## Observability

Suggested instrumentation (some not yet implemented):

- **Logs**: `kubectl logs -n litellm -l app.kubernetes.io/name=litellm`
- **Metrics**: Add sidecar or embed Prometheus exporter (LiteLLM telemetry disabled currently)
- **Tracing**: Wrap API gateway with OpenTelemetry collector (future enhancement)

Health checks:

```bash
# Basic liveness simulation
curl -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4o","messages":[{"role":"user","content":"ping"}]}' \
     http://litellm.litellm.svc.cluster.local:4000/v1/chat/completions | jq '.id'
```

---

## Security & Hardening

| Area          | Current                   | Improvement                                    |
| ------------- | ------------------------- | ---------------------------------------------- |
| Master Key    | Stored in secret          | Rotate periodically; audit access              |
| Provider Keys | In single secret          | Split per provider + RBAC restrict             |
| Redis         | No auth in use            | Add password + TLS, limit network access       |
| Transport     | Plain HTTP inside cluster | Add mTLS via service mesh / ingress TLS        |
| Telemetry     | Disabled                  | Re-enable with scrubbed PII if insights needed |
| UI            | Enabled, no note of auth  | Restrict via auth proxy or disable in prod     |

---

## Troubleshooting

### 1. 401 Unauthorized

- Missing / invalid `Authorization: Bearer` header.
- Master key mismatch—re-fetch secret.

### 2. 500 Errors from Provider

- Check provider quota: environment secret values valid?
- Inspect pod logs for upstream error JSON.

### 3. High Latency

- Check Redis connectivity: `redis-cli -h redis.redis.svc.cluster.local PING`.
- Validate Postgres pool usage: look for saturation or connection errors.

### 4. Cache Not Working

- Ensure `cache: true` in both `model_info` and `litellm_settings`.
- Confirm Redis key writes: `redis-cli KEYS 'litellm_cache*' | head`.

### 5. Schema Issues in Postgres

- Ensure schema `litellm` exists or migrations applied (if LiteLLM uses migrations). Create manually if needed.

### 6. Rate Limit Mismatch

- Adjust Redis-based coordination using router settings or add explicit per-model quotas.

### 7. Pod CrashLoopBackOff

- Inspect logs for missing env secret references.
- Ensure secrets are in same namespace (`litellm`).

---

## Future Improvements

- Add Redis AUTH + TLS and rotate credentials.
- Implement Prometheus metrics & dashboards (requests, latency, token usage, cache hit rate).
- Add OpenTelemetry tracing for end-to-end call chains.
- HPA + Vertical Pod Autoscaler (evaluation) for dynamic scaling.
- Canary strategy for upgrading LiteLLM versions.
- Per-provider concurrency + rate limit configs.
- Optionally enable streaming endpoint demos in UI with auth guard.

---

## Quick Reference

| Item                 | Value                                                                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Namespace            | `litellm`                                                                                                                                    |
| Service DNS          | `litellm.litellm.svc.cluster.local`                                                                                                          |
| Port                 | 4000                                                                                                                                         |
| Replicas             | 3                                                                                                                                            |
| DB URL               | `postgresql://litellm:$(DATABASE_PASSWORD)@postgresql-cloudnative-pg-cluster-pooler-rw.postgresql.svc.cluster.local:5432/app?schema=litellm` |
| Redis                | `redis.redis.svc.cluster.local:6379`                                                                                                         |
| Cache TTL            | 86400s                                                                                                                                       |
| Master Key Secret    | `litellm-master-key` / key `LITELLM_MASTER_KEY`                                                                                              |
| Provider Keys Secret | `litellm-provider-keys`                                                                                                                      |
| DB Password Secret   | `postgres` (env key: `DATABASE_PASSWORD`)                                                                                                    |
| Public URL Link      | `https://litellm.api.prod.everycure.org`                                                                                                     |

> Review and adjust if Helm values change or additional providers are added.
