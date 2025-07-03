# GCP Environments


This guide explains how to use development and production GCP environments with Kedro in the matrix project.

## Terminology

### Kedro environments

Refers to kedro environments (e.g. `base`, `test`, `cloud`) as defined by [kedro documentation here](https://docs.kedro.org/en/stable/configuration/configuration_basics.html#configuration-environments).
Primarily, they have an influence on the location and type of inputs and outputs of your data pipeline, controlled by the data catalog.

### GCP environments

It is abstracted away from the user (you don't set it, it's determined automatically). It is used as a shorthand for  GCP project a pipeline is sent to for execution.
As such, gcp environment `prod` refers to GCP project `mtrx-hub-prod-sms` and `dev` to `mtrx-hub-dev-3of`.

### Runtime variables

Not an environment, but a related concept. 
Variables such as `RUNTIME_GCP_BUCKET` refer to the bucket used for pipeline execution.

**Important**: As of the latest update, the GCP project ID is automatically detected from:
1. GCP metadata server (when running in GKE/GCP environments)
2. gcloud CLI configuration (for local development)

Example: When you run a pipeline, the system will:
- Auto-detect your GCP project (e.g., `mtrx-hub-prod-sms` for prod or `mtrx-hub-dev-3of` for dev)
- Set `RUNTIME_GCP_PROJECT_ID` environment variable to this detected value
- Use this for all downstream operations

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

**Auto-Detection**: The system now automatically detects which GCP environment to use based on:
1. GCP metadata server (when running in GKE/GCP environments)
2. Your active gcloud configuration (for local development)

To switch between environments locally:
- For development: `gcloud config set project mtrx-hub-dev-3of`
- For production: `gcloud config set project mtrx-hub-prod-sms`

When using `kedro experiment run` or `kedro run`, the system will automatically use the detected project.


### Environment Variables

**Auto-Detection**: The system now automatically detects and sets most runtime variables. You only need to explicitly set variables that cannot be auto-detected.

For production access (with private datasets), ensure your gcloud is configured correctly:
```bash
# Set your active project to production
gcloud config set project mtrx-hub-prod-sms

# Optional: Set remaining variables in .env if needed
GOOGLE_APPLICATION_CREDENTIALS=/Users/<YOUR_USERNAME>/.config/gcloud/application_default_credentials.json
INCLUDE_PRIVATE_DATASETS=1
```

For development, ensure your gcloud is configured correctly:
```bash
# Set your active project to development  
gcloud config set project mtrx-hub-dev-3of
```

**Note**: The following variables are now auto-detected and set by the system:
- `RUNTIME_GCP_PROJECT_ID` (auto-detected from GCP metadata or gcloud config)
- `RUNTIME_GCP_BUCKET` (auto-determined based on project environment)
- `MLFLOW_URL` (auto-determined based on project environment)

Commands `kedro experiment run` and `kedro run` are environment agnostic.

## Implications of GCP Environment Selection

### Data Access

- **Private Datasets**: In development, private datasets are automatically excluded from pipelines
- **Storage Locations**: Different GCP buckets are used for `dev` vs `prod` gcp-env pipeline outputs
- **Public Datasets**: When running in production gcp-env, public datasets are still ingested from the bucket in the dev GCP project


### CI/CD Considerations

- The CI pipeline is not extended to prod - testing is done in the same way in both environments
- Currently, releases are only triggered in the dev environment (this will change to prod in the future)
