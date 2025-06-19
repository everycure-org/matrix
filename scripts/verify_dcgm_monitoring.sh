#!/bin/bash

# DCGM GPU Monitoring Setup and Verification Script
# This script helps deploy and verify DCGM monitoring for all pods and workflows

set -e

echo "🚀 DCGM GPU Monitoring Setup for ArgoCD"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we're connected to a cluster
if ! kubectl cluster-info &> /dev/null; then
    log_error "Not connected to a Kubernetes cluster"
    exit 1
fi

# Get current context
CONTEXT=$(kubectl config current-context)
log_info "Current kubectl context: $CONTEXT"

echo
echo "📋 Step 1: Checking GPU nodes in the cluster"
echo "============================================="

GPU_NODES=$(kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-l4 --no-headers 2>/dev/null || echo "")
if [ -z "$GPU_NODES" ]; then
    log_warning "No GPU nodes found with label 'cloud.google.com/gke-accelerator=nvidia-l4'"
    log_info "Checking for any nodes with GPU labels..."
    
    ALL_GPU_NODES=$(kubectl get nodes -l cloud.google.com/gke-accelerator --no-headers 2>/dev/null || echo "")
    if [ -z "$ALL_GPU_NODES" ]; then
        log_error "No GPU nodes found in the cluster"
        log_info "Make sure you have GPU node pools created and properly labeled"
        exit 1
    else
        log_info "Found GPU nodes with different accelerator types:"
        kubectl get nodes -l cloud.google.com/gke-accelerator --show-labels | grep gke-accelerator
        log_warning "Update the nodeSelector in values.yaml if using different GPU types"
    fi
else
    log_success "Found GPU nodes:"
    echo "$GPU_NODES"
fi

echo
echo "📋 Step 2: Checking ArgoCD Applications"
echo "======================================="

# Check if ArgoCD namespace exists
if ! kubectl get namespace argocd &> /dev/null; then
    log_error "ArgoCD namespace not found. Is ArgoCD installed?"
    exit 1
fi

# Check DCGM exporter application
log_info "Checking DCGM exporter ArgoCD application..."
if kubectl get application dcgm-exporter -n argocd &> /dev/null; then
    log_success "DCGM exporter application found in ArgoCD"
    
    # Get application status
    APP_STATUS=$(kubectl get application dcgm-exporter -n argocd -o jsonpath='{.status.sync.status}' 2>/dev/null || echo "Unknown")
    HEALTH_STATUS=$(kubectl get application dcgm-exporter -n argocd -o jsonpath='{.status.health.status}' 2>/dev/null || echo "Unknown")
    
    log_info "Application sync status: $APP_STATUS"
    log_info "Application health status: $HEALTH_STATUS"
    
    if [ "$APP_STATUS" != "Synced" ] || [ "$HEALTH_STATUS" != "Healthy" ]; then
        log_warning "Application is not in a healthy state. Checking details..."
        kubectl describe application dcgm-exporter -n argocd
    fi
else
    log_error "DCGM exporter application not found in ArgoCD"
    log_info "You may need to sync the app-of-apps or check the application template"
fi

echo
echo "📋 Step 3: Checking DCGM DaemonSet deployment"
echo "=============================================="

# Check if monitoring namespace exists
if ! kubectl get namespace monitoring &> /dev/null; then
    log_warning "Monitoring namespace not found. It should be created by the ArgoCD application"
else
    log_success "Monitoring namespace exists"
fi

# Check DCGM DaemonSet
if kubectl get daemonset dcgm-exporter -n monitoring &> /dev/null; then
    log_success "DCGM DaemonSet found"
    
    # Get DaemonSet status
    DESIRED=$(kubectl get daemonset dcgm-exporter -n monitoring -o jsonpath='{.status.desiredNumberScheduled}')
    READY=$(kubectl get daemonset dcgm-exporter -n monitoring -o jsonpath='{.status.numberReady}')
    
    log_info "DaemonSet status: $READY/$DESIRED pods ready"
    
    if [ "$READY" != "$DESIRED" ] || [ "$READY" = "0" ]; then
        log_warning "DaemonSet pods not all ready. Checking pod status..."
        kubectl get pods -n monitoring -l app.kubernetes.io/name=dcgm-exporter
        
        log_info "Pod logs (last 10 lines):"
        kubectl logs -n monitoring -l app.kubernetes.io/name=dcgm-exporter --tail=10 || true
    fi
else
    log_error "DCGM DaemonSet not found in monitoring namespace"
fi

echo
echo "📋 Step 4: Checking Prometheus configuration"
echo "============================================"

# Check if Prometheus is running
if kubectl get deployment kube-prometheus-stack-prometheus -n monitoring &> /dev/null; then
    log_success "Prometheus deployment found"
else
    log_warning "Prometheus deployment not found. Check kube-prometheus-stack installation"
fi

# Check ServiceMonitors
log_info "Checking ServiceMonitors..."
SERVICEMONITORS=$(kubectl get servicemonitor -n monitoring | grep dcgm || echo "")
if [ -n "$SERVICEMONITORS" ]; then
    log_success "DCGM ServiceMonitors found:"
    echo "$SERVICEMONITORS"
else
    log_warning "No DCGM ServiceMonitors found"
fi

echo
echo "📋 Step 5: Testing GPU metrics availability"
echo "==========================================="

# Check if we can access metrics from DCGM pods
DCGM_PODS=$(kubectl get pods -n monitoring -l app.kubernetes.io/name=dcgm-exporter --no-headers 2>/dev/null | awk '{print $1}' || echo "")

if [ -n "$DCGM_PODS" ]; then
    for pod in $DCGM_PODS; do
        log_info "Testing metrics from pod: $pod"
        if kubectl exec -n monitoring $pod -- curl -s http://localhost:9400/metrics | head -5 &> /dev/null; then
            log_success "Metrics endpoint accessible from $pod"
            
            # Check for specific metrics
            GPU_TEMP=$(kubectl exec -n monitoring $pod -- curl -s http://localhost:9400/metrics | grep "DCGM_FI_DEV_GPU_TEMP" | head -1 || echo "")
            if [ -n "$GPU_TEMP" ]; then
                log_success "GPU temperature metrics found: $GPU_TEMP"
            else
                log_warning "No GPU temperature metrics found"
            fi
            
            POWER_USAGE=$(kubectl exec -n monitoring $pod -- curl -s http://localhost:9400/metrics | grep "DCGM_FI_DEV_POWER_USAGE" | head -1 || echo "")
            if [ -n "$POWER_USAGE" ]; then
                log_success "Power usage metrics found: $POWER_USAGE"
            else
                log_warning "No power usage metrics found"
            fi
            
            GPU_UTIL=$(kubectl exec -n monitoring $pod -- curl -s http://localhost:9400/metrics | grep "DCGM_FI_DEV_GPU_UTIL" | head -1 || echo "")
            if [ -n "$GPU_UTIL" ]; then
                log_success "GPU utilization metrics found: $GPU_UTIL"
            else
                log_warning "No GPU utilization metrics found (common in GKE)"
            fi
        else
            log_error "Cannot access metrics from $pod"
        fi
        echo
    done
else
    log_warning "No DCGM pods found to test"
fi

echo
echo "📋 Step 6: Testing Argo Workflow DCGM sidecars"
echo "==============================================="

# Check recent workflow pods
WORKFLOW_PODS=$(kubectl get pods -n argo-workflows -l app=kedro-argo --no-headers 2>/dev/null | head -3 | awk '{print $1}' || echo "")

if [ -n "$WORKFLOW_PODS" ]; then
    log_info "Found recent Argo workflow pods with GPU monitoring:"
    for pod in $WORKFLOW_PODS; do
        echo "  - $pod"
        
        # Check if DCGM sidecar is running
        if kubectl get pod $pod -n argo-workflows -o jsonpath='{.spec.containers[*].name}' | grep -q dcgm; then
            log_success "DCGM sidecar found in $pod"
            
            # Try to get metrics from sidecar
            if kubectl exec -n argo-workflows $pod -c nvidia-dcgm-exporter -- curl -s http://localhost:9400/metrics | head -3 &> /dev/null; then
                log_success "DCGM sidecar metrics accessible in $pod"
            else
                log_warning "DCGM sidecar metrics not accessible in $pod (pod may be completed)"
            fi
        else
            log_warning "No DCGM sidecar found in $pod"
        fi
    done
else
    log_info "No recent Argo workflow pods found with kedro-argo label"
    log_info "Run a GPU workflow to test sidecar monitoring"
fi

echo
echo "🔧 Troubleshooting Commands"
echo "==========================="
echo "If you encounter issues, try these commands:"
echo
echo "# Check GPU nodes and their labels:"
echo "kubectl get nodes --show-labels | grep accelerator"
echo
echo "# Check DCGM DaemonSet status:"
echo "kubectl describe daemonset dcgm-exporter -n monitoring"
echo
echo "# Check DCGM pod logs:"
echo "kubectl logs -n monitoring -l app.kubernetes.io/name=dcgm-exporter"
echo
echo "# Port-forward to test metrics directly:"
echo "kubectl port-forward -n monitoring svc/dcgm-exporter 9400:9400"
echo "# Then visit: http://localhost:9400/metrics"
echo
echo "# Check Prometheus targets:"
echo "kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090"
echo "# Then visit: http://localhost:9090/targets"
echo
echo "# Sync ArgoCD application:"
echo "kubectl patch application dcgm-exporter -n argocd --type merge -p '{\"operation\":{\"sync\":{\"revision\":\"HEAD\"}}}'"
echo
echo "# Test GPU stress workflow:"
echo "# Use the stress test Argo workflow template to generate GPU load and verify metrics"

echo
log_success "DCGM GPU monitoring verification complete!"
echo "Check the output above for any issues that need to be addressed."
