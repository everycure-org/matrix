# GCP Billing Labels for Cost Management

This document describes the billing labels applied to all infrastructure resources for cost tracking and optimization.

## Label Structure

### Node Pool Labels (Applied to GKE nodes and compute costs)

| Label | Purpose | Values |
|-------|---------|--------|
| `cost-center` | Primary cost allocation | `infrastructure-management`, `compute-workloads` |
| `workload-category` | Service categorization | `platform-services`, `data-science` |
| `service-tier` | Infrastructure tier | `management`, `compute` |
| `billing-category` | Billing grouping | `infrastructure`, `gpu-compute`, `cpu-compute` |
| `environment` | Environment identifier | `dev`, `prod` |

### Storage Labels (Applied to PVCs and storage costs)

| Label | Purpose | Values |
|-------|---------|--------|
| `cost-center` | Primary cost allocation | `infrastructure-management` |
| `workload-category` | Service categorization | `platform-services` |
| `service-tier` | Infrastructure tier | `management` |
| `billing-category` | Storage type | `infrastructure-storage`, `infrastructure-storage-premium` |
| `component` | Specific service | `prometheus`, `grafana`, `neo4j-database`, etc. |

**Note**: Storage billing categories distinguish between standard (`infrastructure-storage`) and premium SSD storage (`infrastructure-storage-premium`) for accurate cost tracking of high-performance storage requirements.

### Pod Labels (Applied to running workloads)

| Label | Purpose | Values |
|-------|---------|--------|
| `cost-center` | Primary cost allocation | `infrastructure-management` |
| `workload-category` | Service categorization | `platform-services` |
| `service-tier` | Infrastructure tier | `management` |
| `billing-category` | Billing grouping | `infrastructure` |
| `component` | Specific service | `prometheus`, `grafana`, `argo-workflows-server`, etc. |

## Implementation Examples

### Neo4j Database Service

The Neo4j database deployment demonstrates comprehensive billing label implementation:

**Pod Labels**:
```yaml
labels:
  cost-center: infrastructure-management
  workload-category: platform-services
  service-tier: management
  billing-category: infrastructure
  component: neo4j-database
```

**Storage Labels** (for 1.5TB premium SSD volume):
```yaml
volumeClaimTemplate:
  metadata:
    labels:
      cost-center: infrastructure-management
      workload-category: platform-services
      service-tier: management
      billing-category: infrastructure-storage-premium
      component: neo4j-database
```

This labeling strategy enables precise cost tracking of both compute and storage resources for the Neo4j service.

## Cost Tracking Queries

### GCP Billing Console Filters

To track management infrastructure costs in GCP Billing:

1. **All Management Infrastructure**:
   ```
   labels.goog-k8s-node-pool-name="management-nodes" OR
   labels.cost-center="infrastructure-management"
   ```

2. **Management Compute Costs**:
   ```
   service.description="Compute Engine" AND
   labels.goog-k8s-node-pool-name="management-nodes"
   ```

3. **Management Storage Costs**:
   ```
   service.description="Cloud Storage" AND
   labels.cost-center="infrastructure-management"
   ```

4. **By Component**:
   ```
   labels.component="neo4j-database"
   labels.component="prometheus"
   labels.component="grafana"
   ```

5. **Premium Storage Costs**:
   ```
   service.description="Compute Engine" AND
   labels.billing-category="infrastructure-storage-premium"
   ```

### Expected Cost Allocation

After implementation, you should see costs categorized as:

- **Infrastructure Management** (~$500-1500/month)
  - Management node pool compute
  - Management storage (Prometheus, Grafana, MLflow, Neo4j)
    - Standard storage: `infrastructure-storage`
    - Premium SSD storage: `infrastructure-storage-premium` (e.g., Neo4j 1.5TB)
  - Management networking

- **Compute Workloads** (variable, scales to zero)
  - GPU instances for data science workloads
  - CPU instances for data processing
  - Temporary storage for workflows

## Benefits

1. **Clear Cost Attribution**: Separate management vs. compute costs
2. **Budget Planning**: Predictable management costs vs. variable compute costs
3. **Optimization**: Identify which services consume the most resources
4. **Chargeback**: Allocate costs to appropriate teams/projects
5. **Scaling Analysis**: Track cost savings from scale-to-zero implementation

## Monitoring

Set up GCP Budget Alerts for:
- Total infrastructure-management costs > $2000/month
- Individual component costs (e.g., Neo4j storage > $500/month)
- Premium storage costs (`billing-category="infrastructure-storage-premium"`)
- Unexpected growth in management infrastructure costs

## Future Enhancements

- Add team/owner labels for multi-team environments
- Implement automated cost reports by component
- Set up Slack/email alerts for cost thresholds
- Create Grafana dashboards for cost visualization
