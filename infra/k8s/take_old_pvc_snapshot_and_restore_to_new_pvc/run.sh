#!/bin/bash

# This script takes a snapshot of the mlflow PostgreSQ# Verify the new zone
echo "Verifying new PVC zone..."
sleep 10  # Give some time for PV to be properly labeled
NEW_ZONE=$(kubectl get pv $(kubectl get pvc data-mlflow-postgresql-0 -n mlflow -o jsonpath='{.spec.volumeName}') -o yaml | grep "zones/" | sed 's/.*zones\/\([^/]*\)\/.*/\1/' 2>/dev/null || echo "unknown")C and migrates it from us-central1-a to us-central1-c.

set -e  # Exit on any error

# Check if PVC exists and get current zone
echo "Checking current PVC zone..."
if ! kubectl get pvc data-mlflow-postgresql-0 -n mlflow &>/dev/null; then
    echo "❌ PVC data-mlflow-postgresql-0 not found in namespace mlflow"
    exit 1
fi

CURRENT_ZONE=$(kubectl get pv $(kubectl get pvc data-mlflow-postgresql-0 -n mlflow -o jsonpath='{.spec.volumeName}') -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}')
echo "📍 Current PVC is in zone: $CURRENT_ZONE"
echo "🎯 Target zone: us-central1-c"

# Ensure snapshot CRDs and VolumeSnapshotClass are installed
echo "Creating VolumeSnapshotClass..."
kubectl apply -f volume-snapshot-class.yaml

# Take a snapshot of the mlflow PostgreSQL PVC
echo "Creating snapshot of existing PVC..."
kubectl apply -f mlflow-postgres-snapshot.yaml
echo "Waiting for snapshot to be ready..."
kubectl -n mlflow wait --for=condition=ready volumesnapshot mlflow-pg-snap --timeout=10m

# Check if StatefulSet exists
if ! kubectl get statefulset mlflow-postgresql -n mlflow &>/dev/null; then
    echo "❌ StatefulSet mlflow-postgresql not found in namespace mlflow"
    exit 1
fi

# Scale down MLflow to release the PVC
echo "Scaling down MLflow to release PVC..."
kubectl scale statefulset mlflow-postgresql -n mlflow --replicas=0
echo "Waiting for pods to terminate..."
kubectl -n mlflow wait --for=delete pod -l app.kubernetes.io/name=postgresql --timeout=5m || true

# Delete the old PVC (this is safe because we have the snapshot)
echo "Deleting old PVC in zone $CURRENT_ZONE..."
kubectl delete pvc data-mlflow-postgresql-0 -n mlflow

# Wait a moment for cleanup
sleep 5

# Create a new PVC from the snapshot using the original name
echo "Creating zone-specific StorageClass and new PVC in us-central1-c..."
kubectl apply -f storageclass-us-central1-c.yaml
kubectl apply -f mlflow-postgres-restore-pvc.yaml

# Add node selector to force scheduling in us-central1-c before scaling up
echo "Adding node selector to StatefulSet to ensure scheduling in us-central1-c..."
kubectl patch statefulset mlflow-postgresql -n mlflow -p '{"spec":{"template":{"spec":{"nodeSelector":{"topology.kubernetes.io/zone":"us-central1-c"}}}}}'

# Scale MLflow back up
echo "Scaling MLflow back up..."
kubectl scale statefulset mlflow-postgresql -n mlflow --replicas=1

# Wait for PVC to be bound (this happens when pod is scheduled due to WaitForFirstConsumer)
echo "Waiting for PVC to be bound in us-central1-c..."
kubectl -n mlflow wait --for=condition=bound pvc data-mlflow-postgresql-0 --timeout=10m

# Wait for pod to be ready
echo "Waiting for PostgreSQL pod to be ready..."
kubectl -n mlflow wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql --timeout=10m

# Verify the new zone
echo "Verifying new PVC zone..."
sleep 10  # Give some time for PV to be properly labeled
NEW_ZONE=$(kubectl get pv $(kubectl get pvc data-mlflow-postgresql-0 -n mlflow -o jsonpath='{.spec.volumeName}') -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}' 2>/dev/null || echo "unknown")

echo ""
echo "✅ PVC migration complete!"
echo "📍 Original zone: $CURRENT_ZONE"
echo "🎯 New zone: $NEW_ZONE"
if [ "$NEW_ZONE" = "us-central1-c" ]; then
    echo "🎉 Success: MLflow PostgreSQL data migrated to us-central1-c!"
else
    echo "⚠️  Warning: PVC zone verification returned: $NEW_ZONE"
fi