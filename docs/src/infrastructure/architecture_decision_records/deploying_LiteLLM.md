Architecture Decision Record: Adding LiteLLM to Matrix Infrastructure Kubernetes Stack
Date: 04-09-2025
Status: Proposed
Author: Platform Engineering Team

# Context

The MATRIX platform at EveryCure requires a robust LLM gateway to manage multiple AI model providers efficiently. As we expand our AI-driven drug repurposing capabilities using Gemini 2.0 and other LLM providers, we need a unified interface that provides:

- Consistent API interface across multiple LLM providers
- Cost tracking and budget management
- Rate limiting and load balancing
- Caching for improved performance and cost reduction
- Audit logging and compliance
- High availability and scalability

Our current infrastructure runs in MATRIX project and we need a solution that integrates seamlessly with our existing Kubernetes stack and have the ability to lift and shift to another project later down the line.

# Decision

We will deploy LiteLLM Proxy Server as our LLM gateway within our matrix infrastructure Kubernetes stack with the following components:

LiteLLM Proxy deployed via Helm chart
PostgreSQL for persistent storage and user management with PGBouncer for Connection pooling. (Patroni to be set up later when we need Active/Passive Failover and HA)
Redis for caching and distributed state management.
We will create a new **Cloud DNS Private Zone** for internal DNS Resolution.
We will use **ExternalDNS** controller to tie A Records to services.

# Implementation Details

## Components & Deployment Strategy

1. **LiteLLM Core Deployment**

   - Managed via ArgoCD `Application` (`infra/argo/app-of-apps/templates/litellm.yaml`)
   - Sources upstream Helm chart from `https://github.com/BerriAI/litellm.git` (`deploy/charts/litellm-helm`) pinned to `targetRevision: v1.76.1-stable`
   - Runs in namespace `litellm` (auto-created via `CreateNamespace=true` sync option)
   - Replicas: `replicaCount: 3` for basic HA
   - Container image: `ghcr.io/berriai/litellm:main-stable`
   - Service: ClusterIP on port `4000`

2. **Gateway / Routing**

   - Separate ArgoCD `Application` (`litellm-gateway`) with a later sync-wave ("11") to ensure core LiteLLM is healthy first ("10")
   - Exposes HTTP entry via Kubernetes `HTTPRoute` (Gateway API) referencing shared `api-gateway` in namespace `gateway`
   - Public hostname pattern: `litellm.api.<env>.everycure.org` (injected via Helm parameter) and managed by `external-dns` annotation for automated DNS record creation
   - A `HealthCheckPolicy` (GKE specific) ensures GCLB backend health monitoring on `/` port 4000

3. **PostgreSQL**

   - Uses existing CloudNativePG cluster (pooler) – no embedded DB deployment (`db.useExisting: true`, `deployStandalone: false`)
   - Connection URL template (in values):
     `postgresql://litellm:$(DATABASE_PASSWORD)@postgresql-cloudnative-pg-cluster-pooler-rw.postgresql.svc.cluster.local:5432/app?schema=litellm`
   - Schema-level isolation via `?schema=litellm` rather than a separate database
   - Credentials delivered through ExternalSecret -> Kubernetes secret `postgres`

4. **Redis**

   - External in-cluster Redis service `redis.redis.svc.cluster.local:6379` leveraged for:
     - Request routing / rate state (`router_settings.redis_host`)
     - Response + embedding cache (`litellm_settings.cache_params`)
   - Cache configuration: TTL 86400 seconds (1 day), namespace `litellm_cache`, flush batch size 100

5. **Secrets & Key Management**

   - External Secrets (`infra/argo/applications/litellm-gateway/templates/external_secrets.yaml`) provision:
     - `litellm-master-key` (master key for virtual key issuance)
     - `litellm-provider-keys` (OpenAI + Anthropic API keys)
     - `postgres` (DB username/password)
   - Remote secret backend: `ClusterSecretStore gcp-store`
   - Refresh interval: 1h

6. **Model Registry (Initial Set)**

   - OpenAI: `gpt-4o`, `gpt-4o-mini`, `text-embedding-3-large`, `text-embedding-3-small`
   - Anthropic: `claude-3-5-sonnet-20240620`, `claude-4-sonnet-20250514`, `claude-3-5-haiku-20241022`, `claude-3-haiku-20240307`
   - Caching enabled selectively (`model_info.cache: true`) for frequently reused models

7. **Runtime Settings**

   - Retries: `num_retries: 2` (router level)
   - Telemetry disabled initially (`general_settings.telemetry: false`) – will revisit once observability baseline is in place
   - UI enabled for admin operations (`ui.enabled: true`)

8. **Sync & Delivery**
   - ArgoCD automated sync with prune + selfHeal ensures drift correction
   - Ordered deployment by sync-wave enforces dependency: core app before route/DNS exposure

## Rollout Plan

| Phase | Action                                                     | Success Criteria                                 |
| ----- | ---------------------------------------------------------- | ------------------------------------------------ |
| 0     | Apply secrets + DB schema migration (Terraform)            | Secrets synced; schema present                   |
| 1     | Deploy LiteLLM app (wave 10)                               | Pods Ready=3/3; health endpoints responsive      |
| 2     | Deploy gateway route + health check (wave 11)              | External DNS record resolves; health check green |
| 3     | Smoke test API (list models, simple completion, embedding) | All model calls succeed < 2s p95 (initial)       |
| 4     | Enable caching validation                                  | Cache hit ratio > 20% after warm-up              |
| 5     | Onboard first service via virtual key                      | Successful auth & billing metadata recorded      |

## Operations & SRE Considerations

**Scaling:** Horizontal scaling by adjusting `replicaCount`; Redis centralizes shared state. Consider HPA on CPU + custom QPS.

**Failure Modes:**

- Redis outage: falls back to no shared cache/routing state (elevated latency & cost) – document runbook to disable cache params temporarily
- DB outage: key issuance, logging, and user mgmt fail – read-only proxy operations that don't require DB state may still function (confirm upstream behavior)
- Provider rate limit: built-in retry (2 attempts) – consider circuit breaking if elevated 429s > threshold

**Backups:** Rely on existing Postgres backup policy (CloudNativePG). No Redis persistence assumed (ephemeral cache acceptable).

**Disaster Recovery:** Recreate Redis + rehydrate warm cache; re-sync ArgoCD apps; DB schema managed declaratively (Terraform + migrations).

## Security

| Area                | Control                                                               |
| ------------------- | --------------------------------------------------------------------- |
| API Keys            | Stored in GCP Secret Manager, synced via ExternalSecrets              |
| Master Key          | Rotatable via updating secret backend; triggers pod restart on change |
| Network Exposure    | Restricted to gateway-managed hostname; ClusterIP internally          |
| Namespace Isolation | Dedicated `litellm` namespace for policy scoping                      |
| Least Privilege     | DB user limited to `litellm` schema                                   |
| Future Enhancements | Add OPA/Gatekeeper constraints; enable request audit logging          |

## Observability

Initial telemetry disabled. Next steps:

1. Expose Prometheus metrics endpoint (if supported by upstream chart) and add ServiceMonitor
2. Add structured logging config (JSON) for ingestion by existing log pipeline
3. Define SLOs: Availability (99.5%), p95 latency (<1500ms for GPT-4o), error rate (<2%)
4. Add synthetic probe calls for early provider degradation detection

## Open Questions / Future Work

1. Introduce rate limiting & per-team quota enforcement (needs Redis + custom policy layer)
2. Evaluate multi-region active/passive for Postgres (Patroni or Cloud-native failover) when usage justifies
3. Add additional providers (Mistral, Google Gemini) once keys & compliance cleared
4. Evaluate enabling upstream telemetry for usage analytics once privacy review complete
5. Consider Private Service Connect / Internal LB if cross-project consumption grows

# Alternatives Considered

## DNS Resolution (Load Balancer vs other)

A key decision needed to be made was whether to use a GCP Private Load Balancer or not. Our current networking layer already fulfils present L7 and L4 requirements; we can introduce a Private Load Balancer later if/when needs evolve.

# Architecture Diagram

```mermaid
graph TD
    A[Client Applications] --> C[LiteLLM Proxy<br/>Namespace: lite-llm]

    C --> D[PostgreSQL<br/>Persistent Storage<br/>With PGBouncer]
    C --> E[Redis<br/>Caching & State Management]

    F[ExternalDNS Controller] --> G[GCP Cloud DNS Private Zone<br/>Internal DNS Resolution]

    I[Terraform] --> D
    I --> G

    subgraph "Kubernetes Cluster"
        C
        D
        E
        F
    end

    subgraph "GCP Services"
        G
    end

    subgraph "External Providers"
        J[Gemini 2.0]
        K[Other LLM Providers]
    end

    C --> J
    C --> K
```
