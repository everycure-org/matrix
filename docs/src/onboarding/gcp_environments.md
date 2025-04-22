# GCP Environments


This guide explains how to use development and production GCP environments with Kedro in the matrix project.

## Terminology

### Kedro environments

Refers to kedro environments (e.g. `base`, `test`, `cloud`) as defined by [kedro documentation here](https://docs.kedro.org/en/stable/configuration/configuration_basics.html#configuration-environments).
Primarily, they have an influence on the location and type of inputs and outputs of your data pipeline, controlled by the data catalog.

### GCP environments

Takes two values - `dev` and `prod `and determines the GCP project the data pipeline is submitted to (via `kedro experiment run`).
Pipelines running in the production GCP project are able to access private datasets.

Using the `prod` gcp environment when running a pipeline locally (via `kedro run`), will ingest the private dataset from the production GCP project into your machine.

### Runtime variables

Not an environment, but a related concept. 
Variables such as `RUNTIME_GCP_BUCKET` or `RUNTIME_GCP_PROJECT` refer to the bucket or project corresponding to the GCP environment you want to run the pipeline in.

Example 1: You want to run the pipeline in prod, so your `.env` file has:

`RUNTIME_GCP_PROJECT_ID=mtrx-hub-prod-sms`

Example 2: You want to run the pipeline in dev, so your `.env` file has the following (commented out):
`# RUNTIME_GCP_PROJECT_ID=mtrx-hub-prod-sms`

and your `.env.defaults` file has:

`RUNTIME_GCP_PROJECT_ID=mtrx-hub-dev-3of`

## Understanding GCP Environments

In our project, we have two distinct GCP environments:

- **Development (`dev`)**: uses GCP project `mtrx-hub-dev-3of`
- **Production (`prod`)**: uses GCP project `mtrx-hub-prod-sms`

These environments are separate from Kedro environments (base, cloud, test, etc.) which control pipeline configurations.

⚠️
    Only administrators have access to the production environment.

The GCP environment controls:

- Which GCP project is used for data processing
- Which storage buckets are accessed
- Whether private datasets are included in pipeline runs
- Security and access controls

⚠️
Be aware that production contains strictly confidential, private datasets.
## Using GCP Environments with Kedro

### Setting the GCP Environment

When using `kedro experiment run` you can specify the GCP environment using the `--gcp-env` flag.

```bash
# Run in development environment (default)
kedro experiment run --gcp-env dev --username <your-username> --release-version <version>

# Run in production environment (admin-only)
kedro experiment run --gcp-env prod --username <your-username> --release-version <version>
```

### Environment Variables

The GCP environment setting influences several critical environment variables:

```bash
# Development (set in .env.defaults)
RUNTIME_GCP_PROJECT_ID=mtrx-hub-dev-3of
RUNTIME_GCP_BUCKET=mtrx-us-central1-hub-dev-storage
MLFLOW_URL=https://mlflow.platform.dev.everycure.org/

# Production (set in your .env file)
RUNTIME_GCP_PROJECT_ID=mtrx-hub-prod-sms
RUNTIME_GCP_BUCKET=mtrx-us-central1-hub-prod-storage
MLFLOW_URL=https://mlflow.platform.prod.everycure.org/
```


## Implications of GCP Environment Selection

### Data Access

- **Private Datasets**: In development gcp-env, private datasets are automatically excluded from pipelines
- **Storage Locations**: Different GCP buckets are used for `dev` vs `prod` gcp-env pipeline outputs
- **Public Datasets**: When running in production gcp-env, public datasets are still ingested from the bucket in the dev GCP project


### CI/CD Considerations

- The CI pipeline is not extended to prod - testing is done in the same way in both environments
- Currently, releases are only triggered in the dev environment (this will change to prod in the future)

## Changing GCP Environment

### For Regular Kedro Run

For regular `kedro run` commands (not experiments run in Argo), the GCP environment is determined by your `.env` file settings:

1. Create a `.env` file (if it doesn't exist) based on `.env.defaults`
2. Set the appropriate environment variables for your target environment:

```bash
# For production
RUNTIME_GCP_PROJECT_ID=mtrx-hub-prod-sms
RUNTIME_GCP_BUCKET=mtrx-us-central1-hub-prod-storage
MLFLOW_URL=https://mlflow.platform.prod.everycure.org/
```

## Running With Private Datasets

### In Production Pipeline

Private datasets are automatically included when running in the production gcp-env:

```bash
# Submit a pipeline with private datasets in production (admin-only)
kedro experiment --gcp-env prod run -p kg_release
```

Make sure your `.env` file has the correct production settings:

```bash
RUNTIME_GCP_PROJECT_ID=mtrx-hub-prod-sms
RUNTIME_GCP_BUCKET=mtrx-us-central1-hub-prod-storage
MLFLOW_URL=https://mlflow.platform.prod.everycure.org/
```

### Locally With Private Datasets

To run the pipeline locally with private datasets:

1. Use the regular `kedro run` command
2. Add your personal GCP credentials to your `.env` file:

```bash
# Your personal credentials for accessing private datasets
GOOGLE_APPLICATION_CREDENTIALS=/Users/<username>/.config/gcloud/application_default_credentials.json
```

