---
title: Observability Stack
---


# Observability Stack

We use the [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) Helm chart to deploy our observability stack, which includes Prometheus, Grafana, and various exporters. This provides us with a comprehensive monitoring and visualization solution for our Kubernetes cluster.

## Components

### Prometheus

Our Prometheus setup is configured with the following specifications:

- **Retention Period**: 180 days
- **Storage**: 25GB persistent volume using `standard-rwo` storage class
- **Custom Service Monitors**: Configured to monitor:
  - Argo Workflows Server (`/metrics` endpoint)
  - Argo Workflows Controller (`/metrics` endpoint)

### Grafana

Grafana is configured with the following features:

- **Dashboard Management**:
  - Automatic dashboard discovery enabled via sidecar
  - Dashboards are organized in folders
  - Searches for dashboards across all namespaces
  - Supports folder structure from files

## Accessing the Stack

- **Grafana**: Available at `grafana.platform.dev.everycure.org` (protected by IAP)
- **Prometheus**: Available at `prometheus.platform.dev.everycure.org` (protected by IAP)

## Adding Custom Monitoring

### Adding New Service Monitors

To monitor additional services, add new ServiceMonitor configurations under the `additionalServiceMonitors` section in the Helm values. Example:

```yaml
additionalServiceMonitors:
  - name: my-service
    selector:
      matchLabels:
        app.kubernetes.io/name: my-service
    namespaceSelector:
      matchNames:
        - my-namespace
    endpoints:
      - port: metrics
        path: /metrics
```

### Adding Custom Dashboards

1. Create your dashboard in Grafana
2. Export it as JSON
3. Add it to the appropriate folder in your repository
4. Label it with:
   - `grafana_dashboard: "1"`
   - Use `grafana_folder` annotation to specify the folder
