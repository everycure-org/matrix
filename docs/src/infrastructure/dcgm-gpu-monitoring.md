# DCGM GPU Monitoring Setup for GKE with Workflow Attribution

This setup provides comprehensive GPU monitoring for all Argo workflows using NVIDIA DCGM exporter, following Google Cloud's recommended two-DaemonSet architecture for optimal GPU process monitoring and container attribution. **Updated with robust workflow attribution using kube-state-metrics join queries.**

## Architecture Overview

The monitoring solution follows Google Cloud's best practices and consists of:

1. **DCGM Host Engine DaemonSet**: Provides privileged GPU access and process visibility
2. **DCGM Exporter DaemonSet**: Connects to host engine and exposes Prometheus metrics
3. **kube-state-metrics**: Exposes pod labels including `workflow_name` for metric joins
4. **Prometheus Integration**: Scrapes metrics via ServiceMonitor with workflow attribution
5. **Grafana Dashboard**: Visualizes GPU, CPU, and memory metrics with workflow attribution

This approach provides **robust workflow attribution** without relying on regex parsing of pod names, using Prometheus join queries to combine container metrics with workflow metadata.

## Why Two DaemonSets?

Google Cloud recommends the two-DaemonSet approach because:

- **Process Visibility**: Only the privileged host engine can see GPU processes across all containers
- **Container Attribution**: Enables proper mapping of GPU usage to specific pods/containers/namespaces
- **Security Isolation**: Separates privileged GPU access from metric export functionality
- **GKE Compatibility**: Works around GKE's security restrictions on GPU access

Single-container (embedded) approaches fail to provide accurate GPU utilization metrics in containerized environments.

## Components

### 1. DCGM Host Engine DaemonSet

**Location**: `infra/argo/applications/dcgm-exporter/templates/daemonset.yaml`

```yaml
# Privileged container for GPU process access
image: nvcr.io/nvidia/cloud-native/dcgm:3.3.0-1-ubuntu22.04
command: ["nv-hostengine", "-n", "-b", "ALL"]
port: 5555 (DCGM communication)
```

**Key Features**:
- Runs privileged with full GPU driver access
- Provides DCGM host engine on port 5555
- Monitors all GPU processes across containers
- Essential for process-level attribution

### 2. DCGM Exporter DaemonSet

**Location**: `infra/argo/applications/dcgm-exporter/templates/daemonset.yaml`

```yaml
# Metrics exporter connecting to host engine
image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0-3.2.0-ubuntu22.04
args: ["dcgm-exporter", "--remote-hostengine-info", "127.0.0.1:5555", "--collectors", "/etc/dcgm-exporter/counters.csv"]
port: 9400 (Prometheus metrics)
```

**Key Features**:
- Connects to local host engine via 127.0.0.1:5555
- Exports Prometheus-compatible metrics
- Uses Google's recommended metrics configuration
- Provides container/pod attribution

### 3. Prometheus Integration with Workflow Attribution

**ServiceMonitor Configuration** (Updated):
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
  labels:
    release: kube-prometheus-stack  # Required for Prometheus discovery
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: dcgm-exporter
  endpoints:
  - port: metrics
    interval: 15s
    scrapeTimeout: 10s
```

**kube-state-metrics Configuration** (Added for workflow attribution):
```yaml
kube-state-metrics:
  metricLabelsAllowlist:
    - pods=[*]  # Expose all pod labels including workflow_name
  collectors:
    - pods
  extraArgs:
    - --metric-labels-allowlist=pods=[*]
```

This configuration enables `kube_pod_labels` metrics that include workflow metadata:
```
kube_pod_labels{
  namespace="argo-workflows",
  pod="gpu-run-abc123",
  label_workflow_name="gpu-run-abc123",
  label_test_type="gpu-stress-test",
  label_app="kedro-argo"
} 1
```

## Workflow Attribution Approach

### Problem with Traditional Approaches
- **cAdvisor limitations**: Container metrics (`container_memory_usage_bytes`, `container_cpu_usage_seconds_total`) are scraped at the kubelet/node level and don't have access to pod labels during scrape time
- **Regex fragility**: Extracting workflow names from pod names using regex is brittle and not maintainable
- **Direct pod labeling**: Attempting to add pod labels to container metrics through relabeling doesn't work for cAdvisor

### Solution: kube-state-metrics Join Queries
Our robust solution uses Prometheus join queries to combine container metrics with workflow metadata:

1. **kube-state-metrics** exposes pod labels as `kube_pod_labels` metrics
2. **Prometheus join queries** combine container metrics with workflow labels
3. **Grafana dashboards** use these joined metrics for workflow attribution

### Workflow Label Configuration
Workflows must include the `workflow_name` label in pod templates:
```yaml
# In Argo workflow templates
metadata:
  labels:
    app: kedro-argo
    workflow_name: "{{workflow.name}}"
    test_type: gpu-stress-test
```

## Available Metrics (Google's Recommended Set)
### Core GPU Metrics
- `DCGM_FI_DEV_GPU_UTIL`: GPU utilization (%) with container attribution
- `DCGM_FI_DEV_GPU_TEMP`: GPU temperature (°C)
- `DCGM_FI_DEV_POWER_USAGE`: Power consumption (W)

### Memory Metrics
- `DCGM_FI_DEV_FB_FREE`: Framebuffer memory free (MiB)
- `DCGM_FI_DEV_FB_USED`: Framebuffer memory used (MiB)
- `DCGM_FI_DEV_FB_TOTAL`: Total framebuffer memory (MB)

### Performance Metrics
- `DCGM_FI_PROF_SM_ACTIVE`: SM active cycles ratio
- `DCGM_FI_PROF_SM_OCCUPANCY`: SM occupancy fraction
- `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`: Tensor pipe activity
- `DCGM_FI_PROF_PIPE_FP32_ACTIVE`: FP32 pipe activity
- `DCGM_FI_PROF_PIPE_FP16_ACTIVE`: FP16 pipe activity

### I/O Metrics
- `DCGM_FI_PROF_PCIE_TX_BYTES`: PCIe TX bytes
- `DCGM_FI_PROF_PCIE_RX_BYTES`: PCIe RX bytes
- `DCGM_FI_PROF_NVLINK_TX_BYTES`: NVLink TX bytes
- `DCGM_FI_PROF_NVLINK_RX_BYTES`: NVLink RX bytes

All metrics include labels for container, namespace, and pod attribution:
```
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-...",container="main",namespace="argo-workflows",pod="workflow-pod-123"} 85
```

### Workflow Attribution Metrics
To get workflow attribution, use Prometheus join queries with `kube_pod_labels`:

```promql
# GPU utilization by workflow
DCGM_FI_DEV_GPU_UTIL{namespace="argo-workflows"} 
* on(namespace, pod) group_left(label_workflow_name) 
kube_pod_labels{label_workflow_name!=""}

# Container memory usage by workflow  
container_memory_usage_bytes{namespace="argo-workflows", container!="POD", container!=""} 
* on(namespace, pod) group_left(label_workflow_name) 
kube_pod_labels{label_workflow_name!=""}

# Container CPU usage by workflow
rate(container_cpu_usage_seconds_total{namespace="argo-workflows", container!="POD", container!=""}[5m]) 
* on(namespace, pod) group_left(label_workflow_name) 
kube_pod_labels{label_workflow_name!=""}
```

These queries result in metrics with workflow labels:
```
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-...",container="main",namespace="argo-workflows",pod="gpu-run-abc123",label_workflow_name="gpu-run-abc123"} 85
```

## Deployment

### Prerequisites

1. **GKE Cluster** with GPU node pools
2. **ArgoCD** installed and configured
3. **kube-prometheus-stack** deployed with `release: kube-prometheus-stack` label

### Deployment via ArgoCD

The DCGM exporter is deployed via ArgoCD using the app-of-apps pattern:

```bash
# Configuration location
ls infra/argo/applications/dcgm-exporter/
├── Chart.yaml           # Helm chart metadata
├── values.yaml          # Configuration values
├── templates/
│   ├── daemonset.yaml   # Two DaemonSets + Service + ServiceMonitor
│   └── configmap.yaml   # Google's metrics configuration

# Deployment managed by app-of-apps
cat infra/argo/app-of-apps/templates/dcgm-exporter.yaml
```

### Manual Sync

```bash
# Sync dcgm-exporter application
kubectl patch application dcgm-exporter -n argocd --type merge -p '{"operation":{"sync":{"revision":"HEAD"}}}'

# Sync kube-prometheus-stack (for kube-state-metrics configuration)
kubectl patch application kube-prometheus-stack -n argocd --type merge -p '{"operation":{"sync":{"revision":"HEAD"}}}'
```

## Verification

### Check Pod Status

```bash
# Verify both DaemonSets are running
kubectl get pods -n monitoring -l app.kubernetes.io/name=dcgm-exporter

# Expected output:
NAME                             READY   STATUS    RESTARTS   AGE
dcgm-exporter-xxxxx              1/1     Running   0          5m   # Exporter pods
dcgm-exporter-hostengine-xxxxx   1/1     Running   0          5m   # Host engine pods
```

### Check Logs

```bash
# Host engine logs
kubectl logs -n monitoring -l component=hostengine
# Expected: "Started host engine version 3.3.0 using port number: 5555"

# Exporter logs  
kubectl logs -n monitoring dcgm-exporter-xxxxx
# Expected: "Attemping to connect to remote hostengine at 127.0.0.1:5555"
# Expected: "DCGM successfully initialized!"
```

### Test Metrics with Workflow Attribution

```bash
# Port-forward to metrics endpoint
kubectl port-forward -n monitoring svc/dcgm-exporter 9400:9400 &

# Check GPU utilization metrics with container attribution
curl -s http://localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL

# Expected output with container labels:
# DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-...",container="main",namespace="argo-workflows",pod="..."} 75

# Test kube_pod_labels for workflow attribution
kubectl port-forward -n observability svc/kube-prometheus-stack-prometheus 9090:9090 &
# Query: kube_pod_labels{namespace="argo-workflows",label_workflow_name!=""}

# Test join query for workflow attribution
# Query: DCGM_FI_DEV_GPU_UTIL{namespace="argo-workflows"} * on(namespace, pod) group_left(label_workflow_name) kube_pod_labels{label_workflow_name!=""}
```

### Verify Prometheus Scraping

```bash
# Check ServiceMonitor
kubectl get servicemonitor -n monitoring dcgm-exporter -o yaml

# Port-forward to Prometheus
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090 &

# Check targets in Prometheus UI: http://localhost:9090/targets
# Look for dcgm-exporter targets with status "UP"
```

## GKE-Specific Configuration

### Node Affinity

The DaemonSets target GPU nodes using Google's node affinity pattern:

```yaml
nodeAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    nodeSelectorTerms:
    - matchExpressions:
      - key: cloud.google.com/gke-accelerator
        operator: Exists  # Works with any GPU type
```

### Tolerations

```yaml
tolerations:
  - operator: "Exists"  # Tolerates any taint (Google's approach)
```

### Host Volumes

Required for GPU driver access:

```yaml
hostVolumes:
  - name: nvidia-install-dir
    hostPath: /home/kubernetes/bin/nvidia  # GKE GPU driver location
    mountPath: /usr/local/nvidia
```

## Troubleshooting

### Common Issues

#### 1. Host Engine Connection Failed

**Symptoms**: 
```
"Host engine connection invalid/disconnected"
"Not collecting GPU metrics; The requested function was not found"
```

**Solution**: Verify both DaemonSets are running and host engine is accessible:
```bash
kubectl get pods -n monitoring -l component=hostengine
kubectl logs -n monitoring dcgm-exporter-hostengine-xxxxx
```

#### 2. Zero GPU Utilization

**Root Cause**: Single-container embedded approach doesn't provide process attribution.

**Solution**: Ensure using two-DaemonSet approach as configured.

#### 3. Version Compatibility Issues

**Symptoms**: API version mismatch errors

**Solution**: Ensure matching DCGM versions:
- Host Engine: `3.3.0-1-ubuntu22.04`
- Exporter: `3.3.0-3.2.0-ubuntu22.04`

#### 4. Prometheus Not Discovering Targets

**Symptoms**: No targets in Prometheus UI

**Solution**: Verify ServiceMonitor labels:
```bash
kubectl get servicemonitor -n monitoring dcgm-exporter -o yaml | grep -A5 labels:
# Must include: release: kube-prometheus-stack
```

### Debug Commands

```bash
# Check DaemonSet status
kubectl describe daemonset -n monitoring dcgm-exporter
kubectl describe daemonset -n monitoring dcgm-exporter-hostengine

# Test host engine connectivity
kubectl exec -n monitoring dcgm-exporter-xxxxx -- netstat -an | grep 5555

# Check metrics endpoint
kubectl exec -n monitoring dcgm-exporter-xxxxx -- wget -qO- http://localhost:9400/metrics | head -20

# Check node GPU labels
kubectl get nodes -l cloud.google.com/gke-accelerator --show-labels
```

## Grafana Integration with Workflow Attribution

### Importing the Workflow Monitoring Dashboard

1. **Use the pre-built dashboard**:
   ```bash
   # Dashboard location
   cat workflow-monitoring-dashboard.json
   ```

2. **Import in Grafana**:
   - Go to **Dashboards** → **Import**
   - Copy and paste the JSON content
   - Verify Prometheus data source is correct
   - Click **Import**

### Dashboard Features

The dashboard provides comprehensive workflow attribution:

- **Memory Usage by Workflow**: Shows container memory usage with workflow attribution
- **CPU Usage by Workflow**: Shows container CPU usage with workflow attribution  
- **GPU Utilization by Workflow**: Shows GPU compute and memory utilization
- **GPU Memory Usage by Workflow**: Shows GPU memory consumption
- **Current Resource Usage Table**: Tabular view of current resource usage
- **Workflow Filter**: Dropdown to select specific workflows or view all

### Accessing GPU Metrics

1. **Port-forward to Grafana**:
   ```bash
   kubectl port-forward -n observability svc/kube-prometheus-stack-grafana 3000:80
   ```

2. **Login**: Use admin credentials from secret

3. **Query Examples for Workflow Attribution**:
   ```promql
   # GPU utilization by workflow
   sum by (label_workflow_name) (
     DCGM_FI_DEV_GPU_UTIL{namespace="argo-workflows"} 
     * on(namespace, pod) group_left(label_workflow_name) 
     kube_pod_labels{label_workflow_name!=""}
   )
   
   # Memory usage by workflow
   sum by (label_workflow_name) (
     container_memory_usage_bytes{namespace="argo-workflows", container!="POD", container!=""} 
     * on(namespace, pod) group_left(label_workflow_name) 
     kube_pod_labels{label_workflow_name!=""}
   )
   
   # CPU usage by workflow
   sum by (label_workflow_name) (
     rate(container_cpu_usage_seconds_total{namespace="argo-workflows", container!="POD", container!=""}[5m]) 
     * on(namespace, pod) group_left(label_workflow_name) 
     kube_pod_labels{label_workflow_name!=""}
   )
   ```

4. **Pod-level granularity** (shows individual pods within workflows):
   ```promql
   # Memory usage by workflow and pod
   sum by (label_workflow_name, pod) (
     container_memory_usage_bytes{namespace="argo-workflows", container!="POD", container!=""} 
     * on(namespace, pod) group_left(label_workflow_name) 
     kube_pod_labels{label_workflow_name!=""}
   )
   ```

## Monitoring Argo Workflows with Robust Attribution

### Workflow Requirements

Ensure workflow pod templates include the `workflow_name` label:

```yaml
# In Argo workflow templates (argo_wf_spec.tmpl)
metadata:
  labels:
    app: kedro-argo
    workflow_name: "{{workflow.name}}"
    test_type: gpu-stress-test
```

### GPU Usage Attribution with Join Queries

**Basic GPU metrics by workflow**:
```promql
# GPU utilization for specific workflow
sum by (label_workflow_name) (
  DCGM_FI_DEV_GPU_UTIL{namespace="argo-workflows"} 
  * on(namespace, pod) group_left(label_workflow_name) 
  kube_pod_labels{label_workflow_name=~"gpu-run-.*"}
)

# Power consumption by workflow
sum by (label_workflow_name) (
  DCGM_FI_DEV_POWER_USAGE{namespace="argo-workflows"} 
  * on(namespace, pod) group_left(label_workflow_name) 
  kube_pod_labels{label_workflow_name!=""}
)
```

**Container resource attribution**:
```promql
# Container memory usage by workflow
sum by (label_workflow_name) (
  container_memory_usage_bytes{namespace="argo-workflows", container!="POD", container!=""} 
  * on(namespace, pod) group_left(label_workflow_name) 
  kube_pod_labels{label_workflow_name!=""}
)

# Container CPU usage by workflow
sum by (label_workflow_name) (
  rate(container_cpu_usage_seconds_total{namespace="argo-workflows", container!="POD", container!=""}[5m]) 
  * on(namespace, pod) group_left(label_workflow_name) 
  kube_pod_labels{label_workflow_name!=""}
)
```

### Workflow Pod Visibility with Full Attribution

Join queries provide complete workflow metadata:
```
# Resulting metrics include workflow labels:
DCGM_FI_DEV_GPU_UTIL{
  gpu="0",
  UUID="GPU-a417d268-4f39-29be-5b03-4aa9d6ca052c",
  container="main",
  namespace="argo-workflows", 
  pod="gpu-run-abc123",
  label_workflow_name="gpu-run-abc123",
  label_test_type="gpu-stress-test",
  label_app="kedro-argo"
} 85

container_memory_usage_bytes{
  container="main",
  namespace="argo-workflows",
  pod="gpu-run-abc123", 
  label_workflow_name="gpu-run-abc123",
  label_test_type="gpu-stress-test"
} 2147483648
```

### Benefits of Join Query Approach

✅ **Maintainable**: No regex parsing of pod names  
✅ **Robust**: Works with any workflow naming convention  
✅ **Comprehensive**: Covers GPU, CPU, memory, and disk metrics  
✅ **GitOps-compatible**: Configuration managed through ArgoCD  
✅ **Multi-label support**: Access to all workflow labels (test_type, app, etc.)  
✅ **Pod-level granularity**: Can drill down to individual pods within workflows

## References

- [Google Cloud DCGM Documentation](https://cloud.google.com/stackdriver/docs/managed-prometheus/exporters/nvidia-dcgm)
- [NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)
- [DCGM API Field Reference](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html)
- [Prometheus Join Queries](https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching)
- [kube-state-metrics Configuration](https://github.com/kubernetes/kube-state-metrics/blob/main/docs/cli-arguments.md)
