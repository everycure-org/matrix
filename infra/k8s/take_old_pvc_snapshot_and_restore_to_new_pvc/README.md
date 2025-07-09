# MLflow PostgreSQL PVC Zone Migration

## Background

During a cost-cutting initiative, we consolidated all management workloads (ArgoCD, MLFlow) onto a single GKE node pool to optimize resource utilization. However, this change exposed a critical zone mismatch issue that only manifested in production environment.

### The Problem

The MLflow PostgreSQL PVC was originally created in zone `us-central1-a`, but after the infrastructure consolidation, all management workloads (including MLflow) were constrained to run on nodes in zone `us-central1-c`. This created a zone affinity conflict:

- **PostgreSQL PVC**: Located in `us-central1-a`
- **Target Node Pool**: Only available in `us-central1-c`
- **Result**: PostgreSQL pod stuck in `Pending` state, unable to schedule

### Why This Only Happened in Production

This issue was specific to production because:

1. **Regional vs Zonal PVCs**: Development environments often use regional PVCs that can attach to nodes in any zone within the region
2. **Node Pool Configuration**: Production has strict zone constraints for workload isolation and cost optimization
3. **Infrastructure Consolidation**: The move to consolidate workloads onto management nodes only occurred in production

### Impact

- MLflow tracking server could not start due to database unavailability
- ML model deployment and experiment tracking was completely down
- Data scientists were unable to access their experiments and models

## Solution: PVC Zone Migration

We implemented a safe PVC migration process using Kubernetes VolumeSnapshots to move the PostgreSQL data from `us-central1-a` to `us-central1-c`.

### Migration Process

1. **Create VolumeSnapshot**: Take a snapshot of the existing PVC while preserving data integrity
2. **Scale Down**: Safely stop the PostgreSQL StatefulSet to release the PVC
3. **Delete Original PVC**: Remove the PVC in the wrong zone (safe because we have the snapshot)
4. **Create New PVC**: Restore from snapshot in the target zone using a zone-specific StorageClass
5. **Add Node Selector**: Ensure the StatefulSet pods are scheduled in the correct zone
6. **Scale Up**: Restart the PostgreSQL service with the migrated data

### Key Technical Insights

#### StorageClass Configuration

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard-us-central1-c
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-balanced
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

**Important**: GKE's CSI driver does not support explicit zone parameters (`zone: us-central1-c`). Instead, zone placement is controlled through:
- `volumeBindingMode: WaitForFirstConsumer` - PVC waits for pod scheduling
- Node selectors on the StatefulSet to force scheduling in the target zone

#### Node Selector for Zone Affinity

```bash
kubectl patch statefulset mlflow-postgresql -n mlflow -p '{"spec":{"template":{"spec":{"nodeSelector":{"topology.kubernetes.io/zone":"us-central1-c"}}}}}'
```

This ensures the PostgreSQL pod is scheduled on nodes in `us-central1-c`, which then causes the PVC to be created in the same zone due to `WaitForFirstConsumer` binding.

## Files in this Directory

- `volume-snapshot-class.yaml` - VolumeSnapshotClass for creating snapshots
- `mlflow-postgres-snapshot.yaml` - VolumeSnapshot of the original PVC
- `storageclass-us-central1-c.yaml` - Zone-specific StorageClass for the new PVC
- `mlflow-postgres-restore-pvc.yaml` - New PVC created from the snapshot
- `run.sh` - Automated migration script with error handling and verification

## Lessons Learned

1. **Zone Affinity Planning**: Always consider zone placement when designing PVC strategies
2. **Infrastructure Changes**: Test zone constraints in staging environments that mirror production
3. **StatefulSet Dependencies**: Understand how StatefulSets, PVCs, and node placement interact
4. **GKE CSI Behavior**: Zone placement is controlled by pod scheduling, not StorageClass parameters
5. **Backup Strategy**: VolumeSnapshots provide a reliable way to migrate data between zones

## Prevention

To prevent similar issues in the future:

1. **Use Regional PVCs** when workloads may move between zones
2. **Test Infrastructure Changes** in staging environments with similar zone constraints
3. **Monitor PVC-Node Affinity** during infrastructure changes
4. **Document Zone Dependencies** for all stateful workloads

## Verification Commands

```bash
# Check pod status
kubectl get pods -n mlflow

# Verify PVC zone
kubectl get pv $(kubectl get pvc data-mlflow-postgresql-0 -n mlflow -o jsonpath='{.spec.volumeName}') -o yaml | grep "zones/"

# Check PostgreSQL logs
kubectl logs mlflow-postgresql-0 -n mlflow

# Test database connectivity
kubectl exec -it mlflow-postgresql-0 -n mlflow -- psql -U bn_mlflow -d bitnami_mlflow -c "SELECT version();"
```

## Recovery Time

- **Total Downtime**: ~15 minutes
- **Snapshot Creation**: ~2 minutes
- **Data Migration**: ~5 minutes
- **Service Restart**: ~3 minutes
- **Troubleshooting**: ~5 minutes

The quick recovery was possible due to:
- Small database size (8GB)
- Automated script with proper error handling
- Pre-existing VolumeSnapshot infrastructure
