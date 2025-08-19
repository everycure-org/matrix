# Spot Instance Implementation for Matrix Pipeline Infrastructure

## Overview

This document outlines the comprehensive changes made to implement Google Cloud Platform (GCP) Spot instances across the Matrix pipeline infrastructure. The implementation focuses on cost optimization while maintaining high availability and reliability for pipeline workloads.

## Executive Summary

**Cost Impact**: Up to 80% cost reduction on compute resources by leveraging GCP Spot instances
**Availability**: Graceful fallback to regular instances ensures pipeline reliability
**Scope**: Full coverage of all pipeline workloads (CPU and GPU compute)

## Changes Made

### 1. GKE Infrastructure Changes (`infra/modules/stacks/compute_cluster/gke.tf`)

#### 1.1 New Spot Node Pools

**N2D Highmem Spot Node Pools**
```terraform
n2d_spot_node_pools = [for size in [8, 16, 32, 48, 64] : {
  name           = "n2d-highmem-${size}-spot-nodes"
  machine_type   = "n2d-highmem-${size}"
  spot           = true
  max_count      = 20  # Higher than regular pools for availability
  # ... other configuration
}]
```

**GPU Spot Node Pools**
```terraform
gpu_spot_node_pools = [
  {
    name               = "g2-standard-16-l4-spot-nodes"
    machine_type       = "g2-standard-16"
    accelerator_count  = 1
    accelerator_type   = "nvidia-l4"
    spot               = true
    max_count          = 30  # Higher for spot availability
    # ... other configuration
  }
]
```

#### 1.2 Node Pool Configuration Updates

**Increased Max Counts for Spot Pools**
- N2D Spot pools: `max_count = 20` (vs 10 for regular)
- GPU Spot pools: `max_count = 30` (vs 20 for regular)
- Rationale: Higher capacity to handle spot instance preemptions

**Node Pool Integration**
```terraform
node_pools_combined = concat(
  local.n2d_node_pools,
  local.gpu_node_pools,
  local.management_node_pools,
  local.n2d_spot_node_pools,      # Added
  local.gpu_spot_node_pools       # Added
)
```

#### 1.3 Taints and Labels

**Spot Node Taints**
```terraform
"g2-standard-16-l4-spot-nodes" = [
  {
    key    = "nvidia.com/gpu"
    value  = "present"
    effect = "NO_SCHEDULE"
  },
  {
    key    = "spot"
    value  = "true"
    effect = "NO_SCHEDULE"
  },
  {
    key    = "workload"
    value  = "true"
    effect = "NO_SCHEDULE"
  }
]
```

**Node Labels for Cost Tracking**
```terraform
node_pools_labels = {
  for pool in local.node_pools_combined : pool.name => merge(
    {
      spot_node = lookup(pool, "spot", false) ? "true" : "false"
      billing-category = lookup(pool, "spot", false) ? 
        "gpu-compute-spot" : "gpu-compute"  # For GPU pools
        "cpu-compute-spot" : "cpu-compute"  # For CPU pools
    }
  )
}
```

#### 1.4 Fixed Terraform Compatibility Issues

**Solution**: Used `lookup(pool, "spot", false)` for safe attribute access. This would explicitly add the `spot` attribute to all non-spot instances as well.

### 2. Argo Workflow Template Changes (`pipelines/matrix/templates/argo_wf_spec.tmpl`)

#### 2.1 Node Affinity Configuration

**Spot-First Scheduling Strategy**
```yaml
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    # Highest priority: GKE spot instances
    - weight: 100
      preference:
        matchExpressions:
        - key: cloud.google.com/gke-spot
          operator: In
          values: ["true"]
```

#### 2.2 Tolerations for Spot Nodes

**Updated Tolerations**
```yaml
tolerations:
- key: "workload"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
- key: "node-memory-size"
  operator: "Equal"
  value: "large"
  effect: "NoSchedule"
- key: "spot"              # Added for spot instances
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
```

#### 2.3 Template Coverage

**Templates Updated**
- `kedro` template: All CPU-based pipeline tasks
- `neo4j` template: Database-intensive workloads

**Result**: 100% of pipeline workloads now prefer spot instances


#### 2.4 Retry strategy and fallback

To handle short-lived interruptions caused by spot instance reclamations, the Argo workflow template includes a targeted `retryStrategy` that will automatically retry a task when the failure looks like a spot eviction, and avoid retrying for other failure classes (for example, out-of-memory failures).

Behaviour summary:
- Retries are only attempted when the last failure message matches common eviction indicators such as `pod deleted`, `imminent node shutdown` or `node is draining`.
- Retries are explicitly disabled for OOM failures by excluding `lastRetry.exitCode == 137` (the Linux OOM killer exit code).
- The configured limits and backoff are conservative: `limit: 3` with an exponential backoff starting at `duration: "10"` and `factor: "2"`.

Snippet from the template (`pipelines/matrix/templates/argo_wf_spec.tmpl`):

```yaml
retryStrategy:
  limit: 3
  expression: |
    (
      lastRetry.message matches '.*pod deleted.*' ||
      lastRetry.message matches '.*imminent node shutdown.*' ||
      lastRetry.message matches '.*node is draining.*'
    ) && lastRetry.exitCode != 137
  backoff:
    duration: "10"
    factor: "2"
```

Why this helps:
- Spot nodes are reclaimed unpredictably. The retry strategy lets a short interruption be retried automatically so the pod can be rescheduled (potentially onto a non-spot node given the template's preferred-but-not-required spot affinity).
- Explicitly excluding OOM ensures we don't waste retries on failures that are unlikely to succeed by rescheduling.

Where to change it:
- Update the `retryStrategy` block in `pipelines/matrix/templates/argo_wf_spec.tmpl` to adjust match patterns, retry limits, or backoff behaviour.
- If you change node affinity weights or tolerations, consider whether the retry behaviour should be updated as well (e.g., increasing retries if you expect more spot contention).

## Technical Architecture

### Scheduling Flow

1. **First Choice**: Kubernetes scheduler attempts spot node placement
   - Uses `cloud.google.com/gke-spot: true` label
   - Highest weight (100) for maximum preference

2. **Fallback Strategy**: If spot nodes unavailable
   - Secondary preference for custom `spot_node` labels
   - Graceful degradation to regular node pools

3. **Resource Optimization**: 
   - CPU workloads avoid expensive GPU nodes
   - Proper resource isolation maintained

### Cost Optimization Strategy

**Spot Instance Benefits**
- Up to 80% cost reduction compared to regular instances
- Automatic scaling based on availability
- No change to workload reliability due to fallback mechanism

**Billing Categorization**
- `cpu-compute-spot` vs `cpu-compute`
- `gpu-compute-spot` vs `gpu-compute`
- `infrastructure-management` (always regular nodes)

## Implementation Impact

### Immediate Benefits

1. **Cost Reduction**: Significant savings on compute costs
2. **Resource Efficiency**: Better utilization of available capacity
3. **Scalability**: Higher max node counts for spot pools
4. **Monitoring**: Clear cost attribution through labels

### Operational Considerations

1. **Spot Instance Preemption**: 
   - Kubernetes handles rescheduling automatically
   - Higher node pool limits provide buffer capacity
   - Graceful fallback to regular instances

2. **Workload Tolerance**:
   - All pipeline tasks designed to be stateless
   - Fault-tolerant architecture handles interruptions
   - Retry mechanisms already in place

3. **Management Workloads**:
   - ArgoCD, Prometheus, MLflow remain on regular instances
   - Critical infrastructure maintains high availability

## Monitoring and Observability

### Cost Tracking
- Node labels enable granular cost analysis
- Separate billing categories for spot vs regular instances
- Environment and workload type labeling maintained

### Operational Metrics
- Spot instance utilization rates
- Preemption frequency and impact
- Cost savings achieved

## Deployment Notes

### Prerequisites
- GKE cluster with autoscaling enabled
- Proper IAM permissions for spot instance creation
- Monitoring tools configured for cost tracking

### Rollback Strategy
- Remove spot node pools from `node_pools_combined`
- Revert workflow template affinity changes
- All workloads will fall back to regular instances

## Conclusion

The spot instance implementation provides significant cost optimization while maintaining system reliability. The changes are designed to be:

- **Non-disruptive**: Graceful fallback ensures continuity
- **Cost-effective**: Maximizes use of cheaper spot instances  
- **Operationally sound**: Maintains monitoring and troubleshooting capabilities
- **Future-ready**: Architecture supports additional optimizations

This implementation positions the Matrix pipeline infrastructure for substantial cost savings while preserving the high availability and reliability requirements of the data science workloads.