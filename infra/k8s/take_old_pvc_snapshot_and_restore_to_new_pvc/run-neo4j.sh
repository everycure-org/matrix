#!/bin/bash

# This script takes a snapshot of the neo4j PVC and migrates it from us-central1-a to us-central1-c.

set -e  # Exit on any error

echo "🔧 Starting Neo4j PVC Migration from us-central1-a to us-central1-c"
echo "⚠️  WARNING: This is a 1500Gi volume - migration will take significantly longer than MLflow!"

# Check if PVC exists and get current zone
echo "Checking current PVC zone..."
if ! kubectl get pvc data-neo4j-0 -n neo4j &>/dev/null; then
    echo "❌ PVC data-neo4j-0 not found in namespace neo4j"
    exit 1
fi

CURRENT_ZONE=$(kubectl get pv $(kubectl get pvc data-neo4j-0 -n neo4j -o jsonpath='{.spec.volumeName}') -o yaml | grep "zones/" | sed 's/.*zones\/\([^/]*\)\/.*/\1/' 2>/dev/null || echo "unknown")
echo "📍 Current PVC is in zone: $CURRENT_ZONE"
echo "🎯 Target zone: us-central1-c"

if [ "$CURRENT_ZONE" = "us-central1-c" ]; then
    echo "✅ PVC is already in the target zone us-central1-c"
    exit 0
fi

# Ensure snapshot CRDs and VolumeSnapshotClass are installed
echo "Creating VolumeSnapshotClass..."
kubectl apply -f volume-snapshot-class.yaml

# Take a snapshot of the neo4j PVC
echo "📸 Creating snapshot of existing PVC (this may take several minutes for 1500Gi)..."
kubectl apply -f neo4j-snapshot.yaml
echo "⏳ Waiting for snapshot to be ready... (ETA: 5-15 minutes for large volume)"
kubectl -n neo4j wait --for=condition=ready volumesnapshot neo4j-snap --timeout=30m

# Check if StatefulSet exists
if ! kubectl get statefulset neo4j -n neo4j &>/dev/null; then
    echo "❌ StatefulSet neo4j not found in namespace neo4j"
    exit 1
fi

# Scale down Neo4j to release the PVC
echo "⏸️  Scaling down Neo4j to release PVC..."
kubectl scale statefulset neo4j -n neo4j --replicas=0
echo "⏳ Waiting for pods to terminate..."
kubectl -n neo4j wait --for=delete pod -l app=neo4j --timeout=10m || true

# Delete the old PVC (this is safe because we have the snapshot)
echo "🗑️  Deleting old PVC in zone $CURRENT_ZONE..."
kubectl delete pvc data-neo4j-0 -n neo4j

# Wait a moment for cleanup
sleep 5

# Create zone-specific StorageClass and new PVC from the snapshot
echo "🏗️  Creating zone-specific StorageClass and new PVC in us-central1-c..."
kubectl apply -f storageclass-us-central1-c.yaml
kubectl apply -f neo4j-restore-pvc.yaml

# Add node selector to force scheduling in us-central1-c before scaling up
echo "🎯 Adding node selector to StatefulSet to ensure scheduling in us-central1-c..."
kubectl patch statefulset neo4j -n neo4j -p '{"spec":{"template":{"spec":{"nodeSelector":{"topology.kubernetes.io/zone":"us-central1-c"}}}}}'

# Scale Neo4j back up
echo "▶️  Scaling Neo4j back up..."
kubectl scale statefulset neo4j -n neo4j --replicas=1

# Wait for PVC to be bound (this happens when pod is scheduled due to WaitForFirstConsumer)
echo "⏳ Waiting for PVC to be bound in us-central1-c (may take several minutes for large volume)..."
kubectl -n neo4j wait --for=condition=bound pvc data-neo4j-0 --timeout=20m

# Wait for pod to be ready
echo "⏳ Waiting for Neo4j pod to be ready (may take 10+ minutes for 1500Gi volume attachment and startup)..."
kubectl -n neo4j wait --for=condition=ready pod -l app=neo4j --timeout=30m

# Verify the new zone
echo "🔍 Verifying new PVC zone..."
sleep 10  # Give some time for PV to be properly labeled
NEW_ZONE=$(kubectl get pv $(kubectl get pvc data-neo4j-0 -n neo4j -o jsonpath='{.spec.volumeName}') -o yaml | grep "zones/" | sed 's/.*zones\/\([^/]*\)\/.*/\1/' 2>/dev/null || echo "unknown")

echo ""
echo "✅ Neo4j PVC migration complete!"
echo "📍 Original zone: $CURRENT_ZONE"
echo "🎯 New zone: $NEW_ZONE"
if [ "$NEW_ZONE" = "us-central1-c" ]; then
    echo "🎉 Success: Neo4j data migrated to us-central1-c!"
    echo "💾 Volume size: 1500Gi"
    echo "🚀 Neo4j should now be running successfully"
else
    echo "⚠️  Warning: PVC zone verification returned: $NEW_ZONE"
fi

echo ""
echo "🔍 Verification commands:"
echo "kubectl get pods -n neo4j"
echo "kubectl logs neo4j-0 -n neo4j"
echo "kubectl get pvc -n neo4j"
