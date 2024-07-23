# Plugging into our cloud environment

## Pre-requisites

Ensure to install the `gcloud` command line utilities, this can be done as follows:

```bash
brew install --cask google-cloud-sdk
```

Next, authenticate the client:

```bash
gcloud auth login
```

### Accessing the cluster services

> ⚠️ This section assumes basic knowledge of Kubernetes, cluster technology to deploy Docker containers.

Our platform services run on Kubernetes. We recommend installing `k9s` to interact with the cluster services.

```bash
brew install k9s
```

```bash
TODO Add kubectl config command here
```

To access services on the cluster, launch `k9s` through the command line. The tooling comes with out-of-the-box functionality to setup [port-forwarding](TODO add link). Search for for global cluster resources of `service` type, and hit `shift + f` to activiate port-forwarding.

```bash
k9s
```

### GCP Service account

To correctly leverage the GCP services, you will need a service-account key. You can create a key through the [Google CLI](https://cloud.google.com/storage/docs/gsutil_install) as follows:

> NOTE: This will be provisoned using Terraform and git-crypt in the future.

!!! note

    The below token is time bound and thus this command needs to be re-run regularly, it's only meant for temporary local testing, not for a long-running workload. 

```bash
gcloud config set project mtrx-hub-dev-3of
gcloud iam service-accounts keys create --iam-account=test-gcp@mtrx-hub-dev-3of.iam.gserviceaccount.com  conf/local/service-account.json
```

### Vertex token

To succesfully run embeddings in the `prod` environment, export your GCP access token using the following command:

```bash
export VERTEX_AI_ACCESS_TOKEN=$(gcloud auth print-access-token)
```