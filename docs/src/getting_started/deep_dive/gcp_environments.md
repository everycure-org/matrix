# GCP Environments


This guide explains how to use development and production GCP environments with Kedro in the matrix project.

## Terminology

### Kedro environments

Refers to kedro environments (e.g. `base`, `test`, `cloud`) as defined by [kedro documentation here](https://docs.kedro.org/en/stable/configuration/configuration_basics.html#configuration-environments).
Primarily, they have an influence on the location and type of inputs and outputs of your data pipeline, controlled by the data catalog.

### GCP environments

It is abstracted away from the user (you don't set it, it's determined automatically). It is used as a shorthand for  GCP project a pipeline is sent to for execution.
As such, gcp environment `prod` refers to GCP project `mtrx-hub-prod-sms` and `dev` to `mtrx-hub-dev-3of`. Note that prod environment is only available to a restrictred group of users at the moment.

### Runtime variables

Not an environment, but a related concept. 
Variables such as `RUNTIME_GCP_BUCKET` or `RUNTIME_GCP_PROJECT` refer to the bucket or project corresponding to the GCP project you want to run the pipeline in.

Example 1: You want to run the pipeline in prod, so your `.env` file has:

`RUNTIME_GCP_PROJECT_ID=mtrx-hub-prod-sms`

Example 2: You want to run the pipeline in dev, so your `.env` file has the following (commented out):
`# RUNTIME_GCP_PROJECT_ID=mtrx-hub-prod-sms`

and your `.env.defaults` file has:

`RUNTIME_GCP_PROJECT_ID=mtrx-hub-dev-3of`

## Understanding GCP Environments

In our project, we have two distinct GCP environments (projects):

- **Development (`dev`)**: uses GCP project `mtrx-hub-dev-3of`
- **Production (`prod`)**: uses GCP project `mtrx-hub-prod-sms`

These environments are separate from Kedro environments (base, cloud, test, etc.) which control pipeline configurations.

!!! note 
    Only administrators have access to the production environment.

The GCP environment controls:

- Which GCP project is used for data processing
- Which storage buckets are accessed
- Whether private datasets are available
- Security and access controls

!!! note 
    Be aware that production contains strictly confidential, private datasets.
## Using GCP Environments with Kedro

### Setting the GCP Environment

When using `kedro experiment run` or `kedro run` you can specify the GCP environment using the values in your `.env` file (for `prod`), or have those commented out, in which case `.env.defaults` will be applied, which always point to `dev`.


### Environment Variables

To run pipelines in production (with private datasets), your `.env` file should contain:
```bash
RUNTIME_GCP_PROJECT_ID=mtrx-hub-prod-sms
RUNTIME_GCP_BUCKET=mtrx-us-central1-hub-prod-storage
MLFLOW_URL=https://mlflow.platform.prod.everycure.org/
ARGO_PLATFORM_URL=https://argo.platform.prod.everycure.org
GOOGLE_APPLICATION_CREDENTIALS=/Users/<YOUR_USERNAME>/.config/gcloud/application_default_credentials.json
INCLUDE_PRIVATE_DATASETS=1
```

To run pipelines in development (only public datasets),  the above values in your `.env` should be commented out.

Commands `kedro experiment run` and `kedro run` are environment agnostic.

## Implications of GCP Environment Selection

### Data Access

- **Private Datasets**: In development, private datasets are automatically excluded from pipelines
- **Storage Locations**: Different GCP buckets are used for `dev` vs `prod` gcp-env pipeline outputs
- **Public Datasets**: When running in production gcp-env, public datasets are still ingested from the bucket in the dev GCP project

!!! info
    The CI pipeline is not extended to prod - testing is done in the same way in both environments. Currently, releases are only triggered in the dev environment (this will change to prod in the future)

[Kedro Extensions:](./kedro_extensions.md){ .md-button .md-button--primary }