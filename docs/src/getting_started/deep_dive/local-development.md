
### Accessing the cluster services

!!! info
    This section assumes basic knowledge of Kubernetes, cluster technology to deploy Docker containers.

Our platform services run on Kubernetes. We recommend installing `k9s` to interact with the cluster services.

```bash
brew install k9s
```

```bash
gcloud container clusters get-credentials compute-cluster --region us-central1 
```

To access services on the cluster, launch `k9s` through the command line. The tooling comes with out-of-the-box functionality to setup [port-forwarding](https://thinhdanggroup.github.io/k9s-cli/). Search for for global cluster resources of `service` type, and hit `shift + f` to activiate port-forwarding.

```bash
k9s
```

