# GKE Safe Eviction Configuration

This document contains the commands needed to configure safe eviction annotations for GKE system pods to enable compute nodes to scale to zero.

## Overview

For GKE cluster autoscaler to scale compute node pools to zero, all pods running on compute nodes must have the `cluster-autoscaler.kubernetes.io/safe-to-evict: "true"` annotation. This includes system DaemonSets and Deployments managed by GKE.

## System Pods Safe Eviction Commands

### Core System Components

These commands add the safe-to-evict annotation to system components:

```bash
# CoreDNS (kube-dns) - DNS service
kubectl patch deployment kube-dns -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# Kube-proxy - Network proxy on each node
kubectl patch daemonset kube-proxy -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'
```

### GKE Managed System Components

These are typically already configured with safe-to-evict annotations, but can be patched if needed:

```bash
# Filestore CSI driver
kubectl patch daemonset filestore-node -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# Fluent Bit for logging
kubectl patch daemonset fluentbit-gke -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# GKE metrics agent
kubectl patch daemonset gke-metrics-agent -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# Persistent disk CSI driver
kubectl patch daemonset pdcsi-node -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# Google Managed Prometheus collector
kubectl patch daemonset collector -n gmp-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# Event exporter
kubectl patch deployment event-exporter-gke -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# GMP operator
kubectl patch deployment gmp-operator -n gmp-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# Konnectivity agent
kubectl patch deployment konnectivity-agent -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# Konnectivity agent autoscaler
kubectl patch deployment konnectivity-agent-autoscaler -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# DNS autoscaler
kubectl patch deployment kube-dns-autoscaler -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# L7 default backend
kubectl patch deployment l7-default-backend -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# Metrics server
kubectl patch deployment metrics-server -n kube-system -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'

# Kube state metrics (if running on compute nodes)
kubectl patch statefulset kube-state-metrics -n gke-managed-cim -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'
```

### Monitoring Components

For node-exporter and other monitoring components that should only run on compute nodes:

```bash
# Node exporter (should run only on compute nodes with safe-to-evict)
kubectl patch daemonset kube-prometheus-stack-prometheus-node-exporter -n observability -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict":"true"}}}}}'
```

Note: This is automatically annotated through the Helm Chart. However, if it is not, running it one time will solve the problem.

## Restart Commands

After applying the patches, restart the deployments/daemonsets to pick up the new annotations:

```bash
# Restart core DNS
kubectl rollout restart deployment kube-dns -n kube-system

# Restart kube-proxy
kubectl rollout restart daemonset kube-proxy -n kube-system

# Restart other system components if needed
kubectl rollout restart deployment event-exporter-gke -n kube-system
kubectl rollout restart deployment gmp-operator -n gmp-system
kubectl rollout restart deployment konnectivity-agent -n kube-system
kubectl rollout restart deployment konnectivity-agent-autoscaler -n kube-system
kubectl rollout restart deployment kube-dns-autoscaler -n kube-system
kubectl rollout restart deployment l7-default-backend -n kube-system
kubectl rollout restart deployment metrics-server -n kube-system
```

## Cleanup Commands

To clean up completed jobs that don't have safe-to-evict annotations:

```bash
# Clean up completed argo-workflow cleanup jobs
kubectl get jobs -n argo-workflows --field-selector status.successful=1 -o name | xargs kubectl delete -n argo-workflows

# Clean up completed neo4j certificate refresh jobs
kubectl get jobs -n neo4j --field-selector status.successful=1 -o name | xargs kubectl delete -n neo4j
```

## Verification Commands

To verify that pods have the safe-to-evict annotation:

```bash
# Check all pods on a specific compute node
kubectl get pods --all-namespaces --field-selector spec.nodeName=NODE_NAME -o jsonpath='{range .items[*]}{.metadata.namespace}{"\t"}{.metadata.name}{"\t"}{.metadata.annotations.cluster-autoscaler\.kubernetes\.io/safe-to-evict}{"\n"}{end}' | column -t

# Check for nodes marked for deletion
kubectl describe nodes | grep -A 5 -B 2 "ToBeDeletedByClusterAutoscaler\|DeletionCandidateOfClusterAutoscaler"

# Check cluster autoscaler events
kubectl get events --all-namespaces --sort-by='.lastTimestamp' | grep -i "scale\|delete\|autoscaler"
```

## Important Notes

1. **Timing**: After applying patches, the cluster autoscaler may take 10-20 minutes to recognize nodes as safe to evict and scale them down.

1. **System Pod Scheduling**: Some system pods (like CoreDNS, konnectivity-agent) may have multiple replicas that need to be distributed across available nodes for high availability.

1. **Node Taints**: Nodes marked for deletion will have taints like:

   - `DeletionCandidateOfClusterAutoscaler=TIMESTAMP:PreferNoSchedule`
   - `ToBeDeletedByClusterAutoscaler=TIMESTAMP:NoSchedule`

1. **Resource Requirements**: Ensure the management node pool has sufficient resources to handle any system pods that might need to be rescheduled from compute nodes.

## Troubleshooting

If compute nodes are not scaling down:

1. Check for pods without safe-to-evict annotations
1. Look for completed jobs that should be cleaned up
1. Verify that no application workloads are scheduled on compute nodes
1. Check cluster autoscaler logs in GCP Console
1. Ensure node pools have `min_count = 0` in Terraform configuration
1. **Check for orphaned Pod Disruption Budgets (PDBs)**

### Orphaned Pod Disruption Budgets

PDBs left behind by deleted ArgoCD applications can block node eviction even if the associated pods are failing:

```bash
# Check for PDBs that might be blocking eviction
kubectl get pdb --all-namespaces

# Check PDB status - look for DisruptionAllowed: False
kubectl describe pdb PDB_NAME -n NAMESPACE

# If a PDB is orphaned (no corresponding ArgoCD application), delete it
kubectl delete pdb PDB_NAME -n NAMESPACE

# Also clean up any associated failing deployments
kubectl delete deployment DEPLOYMENT_NAME -n NAMESPACE
```

**Warning**: Only delete PDBs for applications that are confirmed to be unused or have been intentionally removed.

## Related Documentation

- [GKE Cluster Autoscaler](https://cloud.google.com/kubernetes-engine/docs/concepts/cluster-autoscaler)
- [Pod Disruption Budgets](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/)
- [Node Pool Management](./node-pool-management.md)
- [Billing Labels](./billing-labels.md)
