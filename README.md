# MATRIX

[![CI pipline](https://github.com/everycure-org/matrix/actions/workflows/matrix-ci.yml/badge.svg?branch=main)](https://github.com/everycure-org/matrix/actions/workflows/matrix-ci.yml)
[![Infrastructure Deploy](https://github.com/everycure-org/matrix/actions/workflows/infra-deploy.yml/badge.svg?branch=infra&event=push)](https://github.com/everycure-org/matrix/actions/workflows/infra-deploy.yml)
[![Documentation Page Deployment](https://github.com/everycure-org/matrix/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/everycure-org/matrix/actions/workflows/docs-deploy.yml)
[![Evidence Dashboard Deployment](https://github.com/everycure-org/matrix/actions/workflows/evidence-deploy.yml/badge.svg)](https://github.com/everycure-org/matrix/actions/workflows/evidence-deploy.yml)

This repo contains the infrastructure for the MATRIX project for drug repurposing, including data science pipelines, documentation and data base configurations.

**Please visit our [Documentation Page](http://docs.dev.everycure.org) for all details, including onboarding instructions and documentation about the pipeline and our infrastructure. 

## Contributing

1. [General background and vision](https://www.notion.so/everycure/Background-Information-and-Vision-References-600ec31c445f46a7987ff88ea8f67665?pvs=4)
2. [Technical onboarding](https://github.com/everycure-org/matrix/issues/new?assignees=&labels=onboarding&projects=&template=onboarding.md&title=%3Cfirstname%3E+%3Clastname%3E)

## Related Projects

- [MATRIX disease list](https://github.com/everycure-org/matrix-disease-list) - Repo to manage the MATRIX disease list.

# MLflow Deployment with ArgoCD

This repository contains the necessary configuration to deploy Bitnami's MLflow Helm chart using ArgoCD.

## Prerequisites

- Kubernetes cluster
- ArgoCD installed in your cluster
- `kubectl` configured to communicate with your cluster
- Appropriate permissions to create resources in your cluster

## Deployment Instructions

1. **Apply the ArgoCD Application manifest:**

   ```bash
   kubectl apply -f mlflow-application.yaml
   ```

2. **Verify the application is created in ArgoCD:**

   ```bash
   kubectl get applications -n argocd
   ```

3. **Check the sync status:**

   ```bash
   kubectl get applications mlflow -n argocd -o jsonpath='{.status.sync.status}'
   ```

## Configuration

The `mlflow-application.yaml` file contains the ArgoCD Application definition that points to the Bitnami MLflow Helm chart. You should customize the following values before applying:

- `spec.source.targetRevision`: Set to a specific chart version if needed
- `spec.destination.namespace`: The namespace where MLflow will be deployed
- Values under `spec.source.helm.values`:
  - `tracking.host`: Set to your domain
  - Database credentials
  - Storage configuration (MinIO or external S3)

## Accessing MLflow

Once deployed, MLflow will be available at the configured host. If you're using a local setup, you may need to set up port forwarding:

```bash
kubectl port-forward svc/mlflow -n mlflow 5000:5000
```

Then access MLflow at: http://localhost:5000

## Customization

To further customize the MLflow deployment, refer to the [Bitnami MLflow chart documentation](https://github.com/bitnami/charts/tree/main/bitnami/mlflow) for all available configuration options.

## Troubleshooting

If the application fails to sync, check the ArgoCD UI or use:

```bash
kubectl describe application mlflow -n argocd
```

For issues with the MLflow pods:

```bash
kubectl get pods -n mlflow
kubectl describe pod <pod-name> -n mlflow
kubectl logs <pod-name> -n mlflow
```

# NOTE: This file was partially generated using AI assistance.
