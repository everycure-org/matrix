---
title: Computer Cluster
---

This document provides an overview and guide for the shared Kubernetes (k8s) cluster running on GCP in our hub and spoke project setup. The cluster resides in the hub project and is accessible by all development teams / working groups from their respective spoke projects through the shared Virtual Private Cloud (VPC).

## Using the cluster for Spark processing

!!! warning

    not yet implemented, volunteers welcome

## Using the cluster for ray processing


!!! warning

    not yet implemented, volunteers welcome

## Access and Permissions

We will provide `roles/container.developer` and `roles/iap.tunnelResourceAccessor` to everyone in the MATRIX project to enable cluster access. 

## Cluster Configuration

### Cluster Setup
The shared Kubernetes cluster is hosted in the hub project. Key configuration details include:

- **Region**: `us-central1` (by default)
- **Node Pools**: Configured with autoscaling enabled
- **Network**: Shared VPC
- **GPU nodes**: Currently not enabled but planning to add these as scale to 0 autoscaling group

### Networking
Networking between the hub and spoke projects leverages a shared VPC:
- **VPC Name**: `matrix-hub-dev-nw`
- **Subnets**: one per region, e.g. EU and US
- **Firewall Rules**: Configured to allow necessary communication between the hub and spoke projects as well as outgoing HTTPs and incoming SSH via GCP IdP

### Secrets

We set all secrets in our `cloud_secrets` terraform module by grabbing our encrypted yaml file from the disk and creating cloud secrets for each of them. 
<!--
## Access and Permissions

### Development Team Access
Development teams in spoke projects are granted access through:
- **IAM Roles**: Roles such as `Kubernetes Engine Developer` provided at the project level
- **Namespace Quotas**: Defined per team to ensure fair resource usage

Configuring access:
1. **IAM Configuration**: Assign appropriate roles to development teams
2. **Namespaces**: Create namespaced resources for isolation

## Usage

### Scheduling Pipeline Runs
Development teams can schedule pipeline runs on the cluster using tools like CronJobs or managed services such as Google Cloud Composer.

### Scaling and Compute

#### Apache Spark
For big data processing, Apache Spark can be deployed on Kubernetes using:
- [Spark Operator](https://github.com/GoogleCloudPlatform/spark-on-k8s-operator)
- Custom Docker images for Spark jobs

#### Ray
For parallel and distributed computing, Ray can be integrated as follows:
- Deploy a Ray cluster within the k8s cluster
- Utilize Ray's autoscaler to dynamically manage compute resources

## Best Practices
- Monitor cluster health with tools like Prometheus and Grafana
- Implement network policies to secure communication
- Ensure proper logging and auditing for compliance

## Troubleshooting
Common issues and solutions:
- **Access Denied**: Verify IAM roles and network configurations
- **Resource Quotas**: Check namespace quotas if jobs fail due to resource limits

## Conclusion
This documentation provides the foundational setup for a shared k8s cluster on GCP using a hub and spoke model. This setup allows several development teams to efficiently run their workloads, leveraging shared compute resources while maintaining isolation and security.

For further information or specific questions, please refer to the additional resources linked below or contact the cloud operations team.

-->
