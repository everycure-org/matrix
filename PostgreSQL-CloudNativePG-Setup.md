# PostgreSQL CloudNativePG Setup Documentation

## What Has Been Done

### 1. PostgreSQL CloudNativePG Operator Setup

- **Operator Installation**: Deployed CloudNativePG operator v0.26.0 via ArgoCD
- **Namespace**: `postgresql` (auto-created by ArgoCD)
- **RBAC**: Full RBAC configuration with cluster roles and service accounts
- **CRDs**: Custom Resource Definitions for PostgreSQL clusters and poolers

### 2. PostgreSQL Cluster Configuration

- **Database Version**: PostgreSQL 17.6
- **High Availability**: 2-instance cluster configuration
- **Storage**: Premium SSD persistent storage with separate WAL storage
- **Monitoring**: Prometheus integration with PodMonitor and PrometheusRule
- **Node Scheduling**: Deployed on management nodes with appropriate tolerations
- **Resource Limits**: 250m CPU, 256Mi memory requests

### 3. PGBouncer Connection Pooling

Two PGBouncer instances configured:

#### Read-Write Pool (`rw`)

- **Type**: Read-write connections
- **Instances**: 1
- **Scheduling**: Management nodes (`workload-type: management`)
- **Service Name**: `postgresql-rw` (in `postgresql` namespace)

#### Read-Only Pool (`ro`)

- **Type**: Read-only connections
- **Instances**: 1
- **Scheduling**: Application nodes (`node-type: application`)
- **Service Name**: `postgresql-ro` (in `postgresql` namespace)

### 4. Backup Configuration

- **Provider**: Google Cloud Storage
- **Bucket**: `everycure-infra-backups`
- **Path**: `/postgresql`
- **Retention**: 7 days
- **Schedule**: Daily backups at 2 AM UTC
- **Method**: Volume snapshots
- **Compression**: bzip2 for both data and WAL

### 5. Security Features

- **Superuser Access**: Enabled
- **TLS**: Enabled by default with CloudNativePG
- **Service Mesh**: Ready for Istio integration

## How to Connect to PostgreSQL

### Connection Details

#### Through PGBouncer (Recommended)

**For Read-Write Operations:**

````bash
# Service endpoint
Host: postgresql-rw.postgresql.svc.cluster.local
Port: 5432

**For Read-Only Operations:**

```bash
# Service endpoint
Host: postgresql-ro.postgresql.svc.cluster.local
Port: 5432

#### Direct Connection to Cluster (Not Recommended for Applications)

```bash
# Primary instance (read-write)
Host: postgresql-rw.postgresql.svc.cluster.local
Port: 5432

# Read replica
Host: postgresql-ro.postgresql.svc.cluster.local
Port: 5432
```

### Getting Database Credentials

The CloudNativePG operator creates default credentials. To retrieve them:

```bash
# Get the cluster secret (contains app user credentials)
kubectl get secret postgresql-app -n postgresql -o yaml

# Get the superuser secret
kubectl get secret postgresql-superuser -n postgresql -o yaml

# Decode credentials (example for app user)
kubectl get secret postgresql-app -n postgresql -o jsonpath='{.data.username}' | base64 -d
kubectl get secret postgresql-app -n postgresql -o jsonpath='{.data.password}' | base64 -d
```

### Connection Examples

#### From Within the Cluster (Pod to Pod)

**Python (psycopg2):**

```python
import psycopg2

# Read-write connection
conn = psycopg2.connect(
    host="postgresql-rw.postgresql.svc.cluster.local",
    port=5432,
    database="app",
    user="app",
    password="<password_from_secret>"
)

# Read-only connection
conn_ro = psycopg2.connect(
    host="postgresql-ro.postgresql.svc.cluster.local",
    port=5432,
    database="app",
    user="app",
    password="<password_from_secret>"
)
```

**Environment Variables Approach:**

```yaml
# In your deployment YAML
env:
  - name: DATABASE_URL
    value: "postgresql://app:$(DB_PASSWORD)@postgresql-rw.postgresql.svc.cluster.local:5432/app"
  - name: DB_PASSWORD
    valueFrom:
      secretKeyRef:
        name: postgresql-app
        key: password
```

#### Port Forwarding for Local Development

```bash
# Forward PGBouncer read-write port
kubectl port-forward -n postgresql svc/postgresql-rw 5432:5432

# Forward PGBouncer read-only port
kubectl port-forward -n postgresql svc/postgresql-ro 5433:5432

### Monitoring and Health Checks

#### Check Cluster Status

```bash
# Get cluster information
kubectl get cluster -n postgresql

# Get cluster detailed status
kubectl describe cluster postgresql -n postgresql

# Check PGBouncer pools
kubectl get pooler -n postgresql
```

#### View Logs

```bash
# PostgreSQL cluster logs
kubectl logs -n postgresql postgresql-1 -c postgres

# PGBouncer logs
kubectl logs -n postgresql postgresql-rw-<pod-id> -c pgbouncer
```

#### Prometheus Metrics

- PostgreSQL metrics: Available via PodMonitor
- PGBouncer metrics: Available via PodMonitor
- Custom PrometheusRule for alerting

### Important Notes

1. **Always use PGBouncer**: Connect through the pooler services (`postgresql-rw`, `postgresql-ro`) rather than directly to PostgreSQL instances
2. **Read/Write Separation**: Use `postgresql-rw` for writes and `postgresql-ro` for read-only operations to optimize performance
3. **SSL/TLS**: Connections are encrypted by default with CloudNativePG
4. **Backup Recovery**: Backups are stored in GCS and can be restored using CloudNativePG recovery procedures
5. **High Availability**: The cluster automatically handles failover between the 2 PostgreSQL instances

### Troubleshooting

#### Common Issues

**PGBouncer Connection Issues:**

```bash
# Check PGBouncer status
kubectl get pooler -n postgresql
kubectl describe pooler postgresql-rw -n postgresql

# Check PGBouncer logs
kubectl logs -n postgresql -l cnpg.io/poolerName=postgresql-rw
```

**Database Connection Issues:**

```bash
# Test connectivity from a debug pod
kubectl run debug --rm -it --image=postgres:17 -- bash

**Check Service Discovery:**

```bash
# Verify services exist
kubectl get svc -n postgresql

# Check endpoints
kubectl get endpoints -n postgresql
```

## Deployment Status

The PostgreSQL cluster with PGBouncer is deployed via ArgoCD and should be automatically managed. Both the operator and cluster applications are configured with automated sync policies.

To verify deployment:

```bash
# Check ArgoCD applications
kubectl get application -n argocd | grep postgresql

# Check cluster health
kubectl get cluster -n postgresql
kubectl get pooler -n postgresql
```
````
