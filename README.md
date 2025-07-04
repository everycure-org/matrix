# MATRIX

[![CI pipline](https://github.com/everycure-org/matrix/actions/workflows/matrix-ci.yml/badge.svg?branch=main)](https://github.com/everycure-org/matrix/actions/workflows/matrix-ci.yml)
[![Infrastructure Deploy](https://github.com/everycure-org/matrix/actions/workflows/infra-deploy.yml/badge.svg?branch=infra&event=push)](https://github.com/everycure-org/matrix/actions/workflows/infra-deploy.yml)
[![Documentation Page Deployment](https://github.com/everycure-org/matrix/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/everycure-org/matrix/actions/workflows/docs-deploy.yml)
[![KG Dashboard Deployment](https://github.com/everycure-org/matrix/actions/workflows/kg-dashboard-deploy.yml/badge.svg)](https://github.com/everycure-org/matrix/actions/workflows/kg-dashboard-deploy.yml)

This repo contains the infrastructure for the MATRIX project for drug repurposing, including data science pipelines, documentation and data base configurations.

**Please visit our [Documentation Page](http://docs.dev.everycure.org) for all details, including onboarding instructions and documentation about the pipeline and our infrastructure. 

## Contributing

1. [General background and vision](https://www.notion.so/everycure/Background-Information-and-Vision-References-600ec31c445f46a7987ff88ea8f67665?pvs=4)
2. [Technical onboarding](https://github.com/everycure-org/matrix/issues/new?assignees=&labels=onboarding&projects=&template=onboarding.md&title=%3Cfirstname%3E+%3Clastname%3E)

## Related Projects

- [MATRIX disease list](https://github.com/everycure-org/matrix-disease-list) - Repo to manage the MATRIX disease list.

## Dynamic Bucket Selection and Data Paths

Matrix now supports dynamic selection of Google Cloud Storage (GCS) buckets for data sources using custom resolvers. Instead of hardcoding bucket paths in configuration files, you can use the following resolvers:

- `get_kg_raw_path_for_source(source_name)`: Automatically selects the correct bucket (dev, prod, or public) for a given data source based on its configuration flags (`is_public`, `is_private`) in `setting.py` file.

This resolver is used in the Kedro `catalog.yml` and other config files to ensure that data is always read from and written to the correct location, depending on the environment and source privacy settings.

**Required environment variables:**
- `DEV_GCS_BUCKET`
- `PROD_GCS_BUCKET`
- `PUBLIC_GCS_BUCKET`

**Example usage in YAML:**
```yaml
filepath: ${get_kg_raw_path_for_source:rtx_kg2}/KGs/rtx_kg2/${globals:data_sources.rtx_kg2.version}/nodes.tsv
```

See the onboarding documentation for more details and examples.
