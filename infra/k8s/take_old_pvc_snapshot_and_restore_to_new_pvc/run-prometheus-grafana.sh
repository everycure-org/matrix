#!/bin/bash

# This script migrates both Prometheus and Grafana PVCs from us-central1-a to us-central1-c.

set -e  # Exit on any error

echo "🔧 Starting Prometheus + Grafana PVC Migration from us-central1-a to us-central1-c"

# Function to check PVC zone
get_pvc_zone() {
    local pvc_name=$1
    local namespace=$2
    kubectl get pv $(kubectl get pvc $pvc_name -n $namespace -o jsonpath='{.spec.volumeName}') -o yaml | grep "zones/" | sed 's/.*zones\/\([^/]*\)\/.*/\1/' 2>/dev/null || echo "unknown"
}

# Check current zones
echo "Checking current PVC zones..."

PROMETHEUS_PVC="prometheus-kube-prometheus-stack-prometheus-db-prometheus-kube-prometheus-stack-prometheus-0"
GRAFANA_PVC="storage-kube-prometheus-stack-grafana-0"
NAMESPACE="observability"

if ! kubectl get pvc $PROMETHEUS_PVC -n $NAMESPACE &>/dev/null; then
    echo "❌ Prometheus PVC not found in namespace $NAMESPACE"
    exit 1
fi

if ! kubectl get pvc $GRAFANA_PVC -n $NAMESPACE &>/dev/null; then
    echo "❌ Grafana PVC not found in namespace $NAMESPACE"
    exit 1
fi

PROMETHEUS_ZONE=$(get_pvc_zone $PROMETHEUS_PVC $NAMESPACE)
GRAFANA_ZONE=$(get_pvc_zone $GRAFANA_PVC $NAMESPACE)

echo "📍 Prometheus PVC (35Gi) is in zone: $PROMETHEUS_ZONE"
echo "📍 Grafana PVC (20Gi) is in zone: $GRAFANA_ZONE"
echo "🎯 Target zone: us-central1-c"

# Ensure snapshot CRDs and VolumeSnapshotClass are installed
echo "Creating VolumeSnapshotClass..."
kubectl apply -f volume-snapshot-class.yaml

# Create snapshots for both
echo "📸 Creating snapshots of both PVCs..."
kubectl apply -f prometheus-snapshot.yaml
kubectl apply -f grafana-snapshot.yaml

echo "⏳ Waiting for Prometheus snapshot to be ready..."
kubectl -n $NAMESPACE wait --for=condition=ready volumesnapshot prometheus-snap --timeout=15m

echo "⏳ Waiting for Grafana snapshot to be ready..."
kubectl -n $NAMESPACE wait --for=condition=ready volumesnapshot grafana-snap --timeout=15m

# Check if StatefulSets exist
echo "Checking StatefulSets..."
if ! kubectl get statefulset prometheus-kube-prometheus-stack-prometheus -n $NAMESPACE &>/dev/null; then
    echo "❌ Prometheus StatefulSet not found"
    exit 1
fi

if ! kubectl get statefulset kube-prometheus-stack-grafana -n $NAMESPACE &>/dev/null; then
    echo "❌ Grafana StatefulSet not found"
    exit 1
fi

# Scale down both services
echo "⏸️  Scaling down Prometheus and Grafana..."
kubectl scale statefulset prometheus-kube-prometheus-stack-prometheus -n $NAMESPACE --replicas=0
kubectl scale statefulset kube-prometheus-stack-grafana -n $NAMESPACE --replicas=0

echo "⏳ Waiting for pods to terminate..."
kubectl -n $NAMESPACE wait --for=delete pod -l app.kubernetes.io/name=prometheus --timeout=10m || true
kubectl -n $NAMESPACE wait --for=delete pod -l app.kubernetes.io/name=grafana --timeout=10m || true

# Delete old PVCs
echo "🗑️  Deleting old PVCs..."
kubectl delete pvc $PROMETHEUS_PVC -n $NAMESPACE
kubectl delete pvc $GRAFANA_PVC -n $NAMESPACE

# Wait for cleanup
sleep 5

# Create new PVCs from snapshots
echo "🏗️  Creating zone-specific StorageClass and new PVCs in us-central1-c..."
kubectl apply -f storageclass-us-central1-c.yaml
kubectl apply -f prometheus-restore-pvc.yaml
kubectl apply -f grafana-restore-pvc.yaml

# Add node selectors to both StatefulSets
echo "🎯 Adding node selectors to StatefulSets for us-central1-c scheduling..."
kubectl patch statefulset prometheus-kube-prometheus-stack-prometheus -n $NAMESPACE -p '{"spec":{"template":{"spec":{"nodeSelector":{"topology.kubernetes.io/zone":"us-central1-c"}}}}}'
kubectl patch statefulset kube-prometheus-stack-grafana -n $NAMESPACE -p '{"spec":{"template":{"spec":{"nodeSelector":{"topology.kubernetes.io/zone":"us-central1-c"}}}}}'

# Scale both services back up
echo "▶️  Scaling Prometheus and Grafana back up..."
kubectl scale statefulset prometheus-kube-prometheus-stack-prometheus -n $NAMESPACE --replicas=1
kubectl scale statefulset kube-prometheus-stack-grafana -n $NAMESPACE --replicas=1

# Wait for PVCs to be bound
echo "⏳ Waiting for PVCs to be bound in us-central1-c..."
kubectl -n $NAMESPACE wait --for=condition=bound pvc $PROMETHEUS_PVC --timeout=15m
kubectl -n $NAMESPACE wait --for=condition=bound pvc $GRAFANA_PVC --timeout=15m

# Wait for pods to be ready
echo "⏳ Waiting for Prometheus pod to be ready..."
kubectl -n $NAMESPACE wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus --timeout=15m

echo "⏳ Waiting for Grafana pod to be ready..."
kubectl -n $NAMESPACE wait --for=condition=ready pod -l app.kubernetes.io/name=grafana --timeout=15m

# Verify new zones
echo "🔍 Verifying new PVC zones..."
sleep 10
NEW_PROMETHEUS_ZONE=$(get_pvc_zone $PROMETHEUS_PVC $NAMESPACE)
NEW_GRAFANA_ZONE=$(get_pvc_zone $GRAFANA_PVC $NAMESPACE)

echo ""
echo "✅ Prometheus + Grafana PVC migration complete!"
echo ""
echo "📊 Prometheus Migration:"
echo "  📍 Original zone: $PROMETHEUS_ZONE"
echo "  🎯 New zone: $NEW_PROMETHEUS_ZONE"
echo "  💾 Volume size: 35Gi"
echo ""
echo "📊 Grafana Migration:"
echo "  📍 Original zone: $GRAFANA_ZONE" 
echo "  🎯 New zone: $NEW_GRAFANA_ZONE"
echo "  💾 Volume size: 20Gi"
echo ""

if [ "$NEW_PROMETHEUS_ZONE" = "us-central1-c" ] && [ "$NEW_GRAFANA_ZONE" = "us-central1-c" ]; then
    echo "🎉 Success: Both Prometheus and Grafana migrated to us-central1-c!"
    echo "📈 Monitoring services should now be running successfully"
else
    echo "⚠️  Warning: Zone verification issues detected"
    echo "   Prometheus zone: $NEW_PROMETHEUS_ZONE"
    echo "   Grafana zone: $NEW_GRAFANA_ZONE"
fi

echo ""
echo "🔍 Verification commands:"
echo "kubectl get pods -n observability"
echo "kubectl logs prometheus-kube-prometheus-stack-prometheus-0 -n observability -c prometheus"
echo "kubectl logs kube-prometheus-stack-grafana-0 -n observability -c grafana"
