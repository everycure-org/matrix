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

### Questions

- Do we need a separate K8s cluster for this?
- Do we need a second Google Project?
- Shall we have _main_ and _dev_ branches?
- Shall we have infra-dev and infra-prod branches?
- If this is about dataset protection, what is the impact if it gets used by unauthorized people?
- What costs can be incurred?
- Is the MVP clear? What are the Minimum Acceptance Criteria?
- How will this be used?
  - Does the public data need to be ²duplicated, in order to facilitate using environment-agnostic code? Or de we add environment checks and modify pipeline.
    1. Higher level envs can access data in lower level envs.
    2. All public data is duplicated.
       CONS
       - (marginally) higher storage cost
       - chance of missing a sync
       PROS
  - Who is the handful of people that needs access to this private data?
    - All of Every Cure staff (sic Pascal)
  - How is the data made available?
    - GCS
    - BigQuery
    - Neo4J
    - MlFlow

    - Argo Workflows processing these private data should not be accessible to unauthorized staff
    - https://argo.platform.dev.everycure.org/workflows/?&phase=Succeeded&phase=Running&phase=Pending&phase=Failed&phase=Error&limit=50
      it's easier to restrict access using namespaces, rather than on per-app level

    → Permissions need to be set on those locations.

  - Should authorization be granted through Google project ids?
    It seems so, as it makes it easier for granting access to MLFlow and Neo4J.

  - Should we create a different yml in the env folder, e.g. cloud-prod?
    Can we inherit from the current conf/cloud?

  - Is there an alternative way to create to a different env?
    Keep cloud and add Elsevier (and its derivates) to the current conf/cloud, and let permissions (and warnings) take care of any authorization exceptions.
    If we go that way, we will need to add error handling to the gcp.py.Neo4J
    We can't solve it with error handling.
    conf/cloud-prod would be nice to inherit from conf/cloud (to be renamed as conf/cloud-dev), because then we only need to add the Elsevier entry to the catalog, not duplicate the entire catalog.
    Instead of erroring out on attempting to access private dataset within an environment that's available to an enduser, private datasets are part of an environment that is inaccessible.


| Issue | Importance | Description | Why is it not covered in tests | Mitigation |
| ------------- | ------------- | ------------- | ------------- | ------------- |

### Technical context


## Decision

### Mitigation: improving the data fabricator

### Short term: run the pipeline on a sample of production data

#### What is the sampling logic?

# Consequences
