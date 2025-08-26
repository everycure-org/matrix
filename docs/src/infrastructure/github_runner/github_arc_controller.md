# GitHub Actions Self-Hosted Runners Deployment Guide

## Overview

This guide sets up GitHub Actions self-hosted runners on the GKE cluster using Actions Runner Controller (ARC) with the following features:

- **Spot Instances**: Runners use cost-effective spot nodes.
- **Ephemeral Runners**: One pod per job for security and reliability.
- **Docker-in-Docker**: Full Docker build support.
- **Auto-scaling**: 0-50 runners based on GitHub Actions queue.
- **Management Node Isolation**: ARC controller runs on dedicated management nodes.

## Deployment Steps

### 1. Set Up GitHub App

1. Create a GitHub App in the `everycure-org` organization
2. Install it on the organization  
3. Get the App ID, Installation ID, and Private Key
4. Add these to the secrets

### 2. Deploy via ArgoCD

The ArgoCD applications have been added to app-of-apps. They will be automatically synced with `git push`

### 3. Verify GitHub Integration

1. Go to https://github.com/everycure-org/matrix/settings/actions/runners
2. You should see "everycure-gha-runners" listed as a runner set
3. Initially shows 0 runners (auto-scales on demand)

## Usage in GitHub Actions

Use this in your `.github/workflows/*.yml` files:

```yaml
name: Example Workflow
on: [push, pull_request]

jobs:
  build:
    runs-on: everycure-gha-runners  # <-- Use this value
    steps:
      - uses: actions/checkout@v4
      
      - name: Build with Docker
        run: |
          docker build -t my-app .
          docker run --rm my-app npm test
```

## Key Features

### Cost Optimization
- **Spot Instances**: ~70% cost savings compared to on-demand
- **Scale to Zero**: No runners when no jobs are queued
- **Right-sizing**: n2d-standard-4 instances (4 vCPUs, 16GB RAM)

### Security & Reliability
- **Ephemeral Runners**: Fresh pod for each job
- **Network Isolation**: Spot nodes with proper taints
- **Resource Limits**: CPU/memory constraints prevent abuse

### Docker Support
- **Docker-in-Docker**: Full Docker daemon per runner
- **Image Caching**: Persistent storage for Docker images
- **Multi-stage Builds**: Full Docker feature support

## Monitoring & Troubleshooting

### Check Runner Status
```bash
# List runner scale sets
kubectl get runnerscaleset -n actions-runner-system

# Check individual runners  
kubectl get pods -n actions-runner-system -l app=gha-runner-scale-set

# View controller logs
kubectl logs -n actions-runner-system deployment/gha-runner-scale-set-controller
```

### GitHub Actions Queue
```bash
# Check if runners are being created for queued jobs
kubectl describe runnerscaleset everycure-gha-runners -n actions-runner-system
```

### Common Issues

1. **No runners scaling**: Check GitHub App permissions and installation
2. **Pod scheduling failures**: Verify node taints/tolerations
3. **Docker issues**: Check DinD container logs in runner pods

### Scaling Configuration

Current limits:
- **Min Runners**: 0 (cost optimization)
- **Max Runners**: 20 (can be increased in values.yaml)
- **Scale Up**: 2x factor, 30s grace period
- **Scale Down**: 60s after job completion

### Cost Tracking

Runners are labeled for billing:
- `billing-category: github-actions-spot`
- `cost-center: compute-workloads`  
- `workload-category: ci-cd`

## Next Steps

1. Test with a simple workflow
2. Monitor costs in GCP billing
3. Adjust scaling parameters based on usage
4. Consider separate runner sets for different workload types
