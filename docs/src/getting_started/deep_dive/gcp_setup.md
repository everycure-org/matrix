# GCP Set-up

Now that you understand how to run different parts of the pipeline and have seen how the data flows through the system, let's set up your environment to work with GCP. This will allow you to access the full range of data and compute resources available in the Matrix platform.

!!! warning
    Note that this section is only applicable for users who are part of Matrix Project & Matrix GCP infrastructure; if you have set up MATRIX codebase on your own cloud platform, these instructions might not be directly applicable for you

## Prerequisites
Make sure you have installed gcloud SDK as mentioned [the first steps](../first_steps/index.md).

!!! Prerequisited
    Before proceeding, ensure you have:
    1. A Google Cloud account with access to the Matrix project (if you don't have that, please [create an onboarding issue](https://github.com/everycure-org/matrix/issues/new?template=onboarding.md))
    2. The Google Cloud SDK installed [see this section of installation](../first_steps/installation.md#cloud-related-tools)
    3. Basic knowledge of Kubernetes and Docker

### Authentication and Access
First, authenticate with Google Cloud:

```bash
gcloud auth login
```

Then, set up your application default credentials:

```bash
gcloud auth application-default login
```
!!! More Resources
    To learn more about [GCP](../../infrastructure/gcp.md) or [Kubernetes](../../infrastructure/kubernetes_cluster.md) within Matrix, go to [Infrastructure section](../../infrastructure/index.md). Note that this part is also a pre-requisite for a [cluster setup](../first_cluster_run/cluster_setup.md)

[Go to GCP Envirnoment Section  :material-skip-next:](../deep_dive/gcp_environments.md){ .md-button .md-button--primary }