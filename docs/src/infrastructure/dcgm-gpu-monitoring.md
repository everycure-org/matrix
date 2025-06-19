# DCGM GPU Monitoring Setup for ArgoCD

This setup provides comprehensive GPU monitoring for all pods and Argo workflows using NVIDIA DCGM exporter.

## Architecture Overview

The monitoring solution consists of:

1. **DCGM DaemonSet**: Runs on all GPU nodes for cluster-wide monitoring
2. **DCGM Sidecars**: Added to Argo workflow pods for per-workflow monitoring
3. **Prometheus Integration**: Collects metrics from both DaemonSet and sidecars
4. **Grafana Dashboard**: Visualizes GPU metrics

## Components

### 1. DCGM DaemonSet (Cluster-wide monitoring)

**Location**: `infra/argo/applications/dcgm-exporter/`

- **DaemonSet**: Deploys DCGM exporter on all GPU nodes
- **Service**: Exposes metrics endpoint
- **ServiceMonitor**: Configures Prometheus scraping
- **ArgoCD Application**: Manages deployment lifecycle

**Key Features**:
- Runs on all nodes with GPU label `cloud.google.com/gke-accelerator=nvidia-l4`
- Provides cluster-wide GPU visibility
- Monitors idle and active GPUs
- Survives workflow pod restarts

### 2. Workflow Sidecars (Per-workflow monitoring)

**Location**: `pipelines/matrix/templates/argo_wf_spec*.tmpl`

- **DCGM Sidecar**: Primary GPU metrics exporter
- **nvidia-smi Sidecar**: Fallback for GKE-restricted metrics

**Key Features**:
- Per-workflow GPU metrics
- Detailed workflow-specific monitoring
- Works with GKE restrictions
- Provides both DCGM and nvidia-smi metrics

### 3. Prometheus Configuration

**Location**: `infra/argo/applications/kube-prometheus-stack/values.yaml`

**ServiceMonitors**:
- `dcgm-exporter-daemonset`: Scrapes DaemonSet metrics
- `dcgm-exporter-pods`: Scrapes generic DCGM pods
- `gpu-metrics-kedro-argo`: Scrapes workflow sidecars

**Additional Scrape Configs**:
- Direct pod scraping for sidecar containers
- Automatic service discovery for GPU pods

### 4. Grafana Dashboard

**Location**: `infra/argo/applications/dcgm-exporter/templates/grafana-dashboard.yaml`

**Metrics Visualized**:
- GPU Temperature
- Power Usage
- GPU Utilization (DCGM + nvidia-smi)
- Memory Utilization
- Memory Usage
- Clock Speeds
- Exporter Status

## Available Metrics

### DCGM Metrics (Primary)
- `DCGM_FI_DEV_GPU_TEMP`: GPU temperature (°C)
- `DCGM_FI_DEV_POWER_USAGE`: Power consumption (W)
- `DCGM_FI_DEV_GPU_UTIL`: GPU utilization (%) - Limited in GKE
- `DCGM_FI_DEV_MEM_COPY_UTIL`: Memory utilization (%)
- `DCGM_FI_DEV_FB_USED`: Memory used (MB)
- `DCGM_FI_DEV_SM_CLOCK`: SM clock speed (MHz)
- `DCGM_FI_DEV_MEM_CLOCK`: Memory clock speed (MHz)

### nvidia-smi Metrics (Fallback)
- `nvidia_smi_gpu_utilization_percent`: GPU utilization (%)
- `nvidia_smi_memory_utilization_percent`: Memory utilization (%)
- `nvidia_smi_memory_used_bytes`: Memory used (bytes)
- `nvidia_smi_temperature_celsius`: GPU temperature (°C)
- `nvidia_smi_power_usage_watts`: Power usage (W)

## Deployment Instructions

### Prerequisites

1. **GPU Node Pool**: Ensure you have GPU nodes in your GKE cluster
2. **ArgoCD**: ArgoCD must be installed and running
3. **Prometheus Stack**: kube-prometheus-stack should be deployed

### Deployment Steps

1. **Deploy via ArgoCD** (Recommended):
   ```bash
   # Run the deployment script
   ./scripts/deploy_dcgm_monitoring.sh
   ```

2. **Manual Deployment**:
   ```bash
   # Sync app-of-apps to pick up dcgm-exporter
   kubectl patch application app-of-apps -n argocd --type merge -p '{"operation":{"sync":{"revision":"HEAD"}}}'
   
   # Sync dcgm-exporter application
   kubectl patch application dcgm-exporter -n argocd --type merge -p '{"operation":{"sync":{"revision":"HEAD"}}}'
   ```

3. **Verification**:
   ```bash
   # Run the verification script
   ./scripts/verify_dcgm_monitoring.sh
   ```

## GKE-Specific Considerations

### Limited Metrics in GKE
Google Cloud restricts some DCGM metrics for security reasons:
- GPU utilization may show as 0
- Some performance counters are disabled
- Temperature and power metrics are usually available

### Node Affinity
The DaemonSet uses node selector `cloud.google.com/gke-accelerator=nvidia-l4`.
Update this in `values.yaml` if using different GPU types:
```yaml
nodeSelector:
  cloud.google.com/gke-accelerator: "nvidia-a100"  # Change as needed
```

### Node Pool Scaling
- DCGM pods will only run when GPU nodes are available
- Pods will start automatically when nodes scale up
- Use cluster autoscaler for dynamic scaling

## Troubleshooting

### Common Issues

1. **No DCGM Pods Running**:
   ```bash
   # Check GPU nodes
   kubectl get nodes -l cloud.google.com/gke-accelerator --show-labels
   
   # Check node selector in values.yaml
   # Ensure it matches your GPU node labels
   ```

2. **Metrics Showing 0**:
   ```bash
   # Check if this is GKE limitation
   kubectl exec -n monitoring <dcgm-pod> -- curl -s http://localhost:9400/metrics | grep UTIL
   
   # Use nvidia-smi exporter as fallback
   # Check workflow sidecar logs
   ```

3. **Prometheus Not Scraping**:
   ```bash
   # Check ServiceMonitors
   kubectl get servicemonitor -n monitoring | grep dcgm
   
   # Check Prometheus targets
   kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
   # Visit: http://localhost:9090/targets
   ```

4. **DaemonSet Not Scheduling**:
   ```bash
   # Check tolerations and node affinity
   kubectl describe daemonset dcgm-exporter -n monitoring
   
   # Check node taints
   kubectl describe nodes | grep -A5 "Taints:"
   ```

### Debugging Commands

```bash
# Check DCGM pod logs
kubectl logs -n monitoring -l app.kubernetes.io/name=dcgm-exporter

# Test metrics manually
kubectl exec -n monitoring <dcgm-pod> -- curl -s http://localhost:9400/metrics

# Check application status
kubectl describe application dcgm-exporter -n argocd

# Check DaemonSet status
kubectl describe daemonset dcgm-exporter -n monitoring

# Port-forward for direct access
kubectl port-forward -n monitoring svc/dcgm-exporter 9400:9400
curl http://localhost:9400/metrics
```

## Monitoring Workflow GPU Usage

### Running GPU Workflows

1. **Use Argo Workflow Templates**: The existing templates include DCGM sidecars
2. **Check Metrics During Execution**:
   ```bash
   # List running workflow pods
   kubectl get pods -n argo-workflows -l app=kedro-argo
   
   # Check metrics from sidecar
   kubectl exec -n argo-workflows <pod> -c nvidia-dcgm-exporter -- curl -s http://localhost:9400/metrics
   ```

3. **Grafana Visualization**: Access the GPU monitoring dashboard in Grafana

### Stress Testing

Use the stress test workflow template to generate GPU load and verify monitoring:
- Template includes both DCGM and nvidia-smi exporters
- Runs GPU stress tests to generate measurable load
- Provides detailed metrics verification

## Grafana Dashboard Access

1. **Port-forward to Grafana**:
   ```bash
   kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80
   ```

2. **Login**: Default credentials are usually `admin/prom-operator`

3. **Find Dashboard**: Look for "GPU Monitoring (DCGM)" in the dashboards

## Metrics Retention

Prometheus retains metrics for 180 days (configured in `values.yaml`).
Adjust the retention period based on your needs:

```yaml
prometheus:
  prometheusSpec:
    retention: 180d  # Adjust as needed
```

## Scaling and Performance

### Resource Usage
- **DCGM DaemonSet**: ~128Mi memory, 100m CPU per node
- **Workflow Sidecars**: ~64Mi memory, 50m CPU per pod
- **Metrics Volume**: ~1KB per metric per scrape interval

### Scaling Considerations
- DaemonSet automatically scales with GPU nodes
- Workflow sidecars scale with workflow pods
- Prometheus storage requirements scale with node count and workflow frequency

## Security

### Privileged Access
DCGM requires privileged access to read GPU metrics:
- DaemonSet runs with `privileged: true`
- Required for hardware-level GPU access
- Limited to monitoring namespace

### Network Policies
Consider implementing network policies to restrict metric endpoint access:
- Allow Prometheus to scrape DCGM pods
- Restrict external access to metric endpoints

## Future Enhancements

### Possible Improvements
1. **Custom Metrics**: Add application-specific GPU metrics
2. **Alerting**: Set up Prometheus alerts for GPU issues
3. **Multi-GPU Support**: Enhanced support for multi-GPU nodes
4. **Historical Analysis**: Long-term GPU usage trending

### Integration Options
1. **MLflow Integration**: Track GPU metrics with ML experiments
2. **Cost Monitoring**: Correlate GPU usage with cloud costs
3. **Autoscaling**: Use GPU metrics for intelligent cluster scaling
