# GCP Set-up

Now that you understand how to run different parts of the pipeline and have seen how the data flows through the system, let's set up your environment to work with GCP. This will allow you to access the full range of data and compute resources available in the Matrix platform.

!!! warning
    Note that this section is only applicable for users who are part of Matrix Project & Matrix GCP infrastructure; if you have set up MATRIX codebase on your own cloud platform, these instructions might not be directly applicable for you

## Prerequisites

!!! Prerequisited
    Before proceeding, ensure you have:
    1. A Google Cloud account with access to the Matrix project (if you don't have that, please [create an onboarding issue](https://github.com/everycure-org/matrix/issues/new?template=onboarding.md))
    2. The Google Cloud SDK installed [see this section of installation](../first_steps/installation.md#cloud-related-tools)
    3. Basic knowledge of Kubernetes and Docker

## Environment Setup

### 1. Authentication and Access

First, authenticate with Google Cloud:

```bash
gcloud auth login
```

Then, set up your application default credentials:

```bash
gcloud auth application-default login
```

### 2. Fetching Required Secrets

The Matrix platform requires certain secrets for operation. Fetch them using:

```bash
make fetch_secrets
```

This will:

- Create a `conf/local` directory
- Fetch the storage service account key
- Fetch the OAuth client secret
- Set appropriate permissions on the secret files

### 3. Accessing Cluster Services

To interact with the Kubernetes cluster:

1. Install `k9s` for cluster management:
```bash
brew install k9s
```

2. Get cluster credentials:
```bash
gcloud container clusters get-credentials compute-cluster --region us-central1
```

3. Launch `k9s` to access services:
```bash
k9s
```

Use `shift + f` in `k9s` to set up port-forwarding for services.

!!! More Resources
    To learn more about [GCP](../../infrastructure/gcp.md) or [Kubernetes](../../infrastructure/kubernetes_cluster.md) within Matrix, go to [Infrastructure section](../../infrastructure/index.md).

[Go to GCP Envirnoment Section  :material-skip-next:](../deep_dive/gcp_environments.md){ .md-button .md-button--primary }