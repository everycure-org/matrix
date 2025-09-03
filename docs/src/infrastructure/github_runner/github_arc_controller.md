# GitHub Actions Self-Hosted Runners Deployment Guide

## Overview

This guide sets up GitHub Actions self-hosted runners on the GKE cluster using Actions Runner Controller (ARC) with the following features:

- **Ephemeral Runners**: One pod per job for security and reliability.
- **Docker-in-Docker**: Full Docker build support.
- **Auto-scaling**: 0-50 runners based on GitHub Actions queue.
- **Management Node Isolation**: ARC controller runs on dedicated management nodes.

## Custom Runner Image

To optimize CI performance and avoid installing dependencies on every job run, we use a custom GitHub runner image that includes pre-installed tools and dependencies.

### Image Location

The custom image is hosted in our Artifact Registry:

```
us-central1-docker.pkg.dev/mtrx-hub-dev-3of/github-runner-images/github-runner:latest
```

### Pre-installed Tools

The custom image extends the official GitHub Actions runner image and includes:

- **Python Environment**:
  - pyenv for Python version management
  - Python 3.11 as the default version
- **Build Tools**:
  - make, build-essential, gcc toolchain
  - SSL, compression, and development libraries
- **Java Runtime**:
  - OpenJDK 17 (JDK and JRE)
- **System Dependencies**:
  - curl, wget, ca-certificates
  - Various development libraries (libssl-dev, zlib1g-dev, etc.)

### Image Source

The Dockerfile and setup scripts are located in:

- `/infra/github-runner-image/Dockerfile`
- `/infra/github-runner-image/setup_pyenv.sh`
- `/infra/github-runner-image/podman-docker-api.sh`

### Image Build Process

The custom image is automatically built and pushed to Artifact Registry using GitHub Actions:

- **Workflow**: `.github/workflows/build_and_upload_image_for_github_runner_set_k8s.yml`
- **Triggers**:
  - Changes to files in `infra/github-runner-image/` directory
  - Changes to the workflow file itself
  - Manual dispatch via GitHub UI
- **Branches**: Builds on `main` and feature branches (currently `nelson/aip-517-optimize-ci-process-to-reduce-runtime-from-50-minutes`)
- **Tags**: Creates both `latest` and SHA-based tags for each build

### Benefits

By using this custom image, we achieve:

- **Faster Job Startup**: No need to install common dependencies on each run
- **Consistent Environment**: All jobs use the same pre-configured base environment
- **Reduced Network Usage**: Dependencies are baked into the image
- **Better Reliability**: Pre-tested tool combinations

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
    runs-on: everycure-gha-runners # <-- Use this value
    steps:
      - uses: actions/checkout@v4

      - name: Build with Docker
        run: |
          docker build -t my-app .
          docker run --rm my-app npm test
```

## Key Features

### Cost Optimization

- **Scale to Zero**: No runners when no jobs are queued
- **Right-sizing**: n2d-standard-4 instances (8 vCPUs, 32GB RAM)

### Security & Reliability

- **Ephemeral Runners**: Fresh pod for each job
- **Network Isolation**: Spot nodes with proper taints
- **Resource Limits**: CPU/memory constraints prevent abuse

### Docker Support

- **Docker-in-Docker**: Full Docker daemon per runner.
- **Image Caching**: Persistent storage for Docker images (Possible. Need extra configuration)
- **Multi-stage Builds**: Full Docker feature support.

## Limitation

Docker Compose usually runs into situation where a docker container gets stuck during initialization. This could be due to:

1. Storage Driver Compatibility & Performance
2. Security and Privileges
3. Networking and Docker Compose Behavior

It is advised to refrain from running docker compose and instead rely on natively running each component.

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
2. **Pod scheduling failures**: Verify node taints/tolerations and ensure spot nodes are available
3. **Docker issues**: Check DinD container logs in runner pods
4. **Cannot connect to Docker daemon**: Ensure DinD sidecar is running properly
5. **Docker build failures**: Check that both runner and DinD containers have adequate resources
6. **Certain containers are stuck when running `docker compose`**: Please do not use `docker compose`.

### Scaling Configuration

Current limits:

- **Min Runners**: 0 (cost optimization)
- **Max Runners**: 50 (can be increased in values.yaml)
- **Scale Up**: 2x factor, 30s grace period
- **Scale Down**: 60s after job completion

### Cost Tracking

Runners are labeled for billing:

- `billing-category: github-actions-spot`
- `cost-center: compute-workloads`
- `workload-category: ci-cd`
