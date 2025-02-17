---
title: Production environment: usage and rollout
status: proposed
---

## Status

Under review

## Context

At the time of writing, Every Cure (EC) and its subcontractors ("subs")
are using a single Google Project under which all of the
infrastructure, like the container storage (GCS), the object store, and
the Kubernetes service, is hosted. Several other services, like Google
Cloud Secret Manager are also hosted here.

This Google Project is called "mtrx-hub-dev", with an id
_mtrx-hub-dev-3of_. As the name suggests, it is a development
environment, which is commonly associated with a place where developers
add new features before being deployed to a production environment,
possibly through staged deployments.

The majority of this set of services is managed using Terraform and Terragrunt,
two popular Infrastructure as Code (IaC) tools. These tools work by looking at
the state of the [infra
branch](https://github.com/everycure-org/matrix/tree/infra) of the matrix
project and deploy the changes from the repo's top-level infra folder.

As part of this environment, EC is hosting data that is public in nature:
anyone can find the information online, the info is free of charge and
unrestricted by licenses.


### Problems

- Datasets that aren't public in nature, which we'll label as sensitive or
  private from here on, will need to be added to Every Cure's set of assets,
  while protecting it with some form of access control so that it can only be
  read by Every Cure staff.
- The state of the infrastructure is coupled to the state of the processing
  code. As an example, some kubernetes pods are sized for processing certain
  datasets. As the processing code evolves, sometimes changes in the pod
  configuration are entered which might break the option of processing data.

### User context


| Issue | Importance | Description | Why is it not covered in tests | Mitigation |
| ------------- | ------------- | ------------- | ------------- | ------------- |

### Technical context


## Decision

### Mitigation: improving the data fabricator

### Short term: run the pipeline on a sample of production data

#### What is the sampling logic?

# Consequences
