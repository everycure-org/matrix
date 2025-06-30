# Cluster Set-up

Now that you understand how to run different parts of the pipeline and have seen how the data flows through the system, let's set up your environment to work with infrastructure and kubernetes cluster. This will allow you to leverage the pipeline in a parallelized and resource-efficient manner.

!!! warning
    Note that this section is only applicable for users who are part of Matrix Project & Matrix GCP infrastructure; if you have set up MATRIX codebase on your own cloud platform, these instructions might not be directly applicable for you

## Tech Stack

To understand the cluster set-up, you will need to have a basic understanding of the following two technologies:

* **Argo Workflows** - Kubernetes-native workflow engine that orchestrates our pipeline execution in production
* **Kubernetes** - Container orchestration platform where our pipeline runs in production environments

## Installation
### Docker Configuration

First, configure Docker to use the Google Container Registry:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### kubectl

Kubectl is a CLI tool we use to interact with our Kubernetes cluster. It is required to submit workflows to the cloud environment.

=== "MacOS"

    ```bash
    brew install kubectl
    # ... test your installation. You should see your kubectl version.
    kubectl version --client
    ```

Once installed, use the gcloud SDK to connect kubectl to the kubernetes cluster. Replace `REGION` and `PROJECT_ID` below with your own values found in GCP.

=== "MacOS"

    ```bash
    gcloud components install gke-gcloud-auth-plugin
    gcloud container clusters get-credentials compute-cluster --region us-central1 --project mtrx-hub-dev-3of
    
    # You can test your installation by running the following command; you should see a list of the cluster's namespaces as output.
    kubectl get namespaces
    ```

### Argo Workflows

[Argo](https://argoproj.github.io/) is our main tool to run jobs in kubernetes. Its CLI tool `argo` is required to submit workflows to the cloud environment.

!!! warning

    Argo Workflows is not the same as ArgoCD. Argo is a family of tools operating on kubernetes. We use both but most people only need to care about Argo Workflows.

=== "MacOS"

    ```bash
    brew install argo
    ```

=== "Linux"

    Check the [official documentation from argo](https://github.com/argoproj/argo-workflows/releases/).

### k9s

Install `k9s` for easier cluster management:

```bash
brew install k9s
```

## Environment Setup

Once you installed the tech stack, you can proceed to the environment setup.

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

1. Get cluster credentials:
```bash
gcloud container clusters get-credentials compute-cluster --region us-central1
```

2. Launch `k9s` to access services:
```bash
k9s
```

Use `shift + f` in `k9s` to set up port-forwarding for services.

[Go to Full Cluster Run Section  :material-skip-next:](../first_cluster_run/full_cluster_run.md){ .md-button .md-button--primary }

