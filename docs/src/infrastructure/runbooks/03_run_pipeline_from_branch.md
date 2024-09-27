---
title: Running an experiment from a branch on the Every Cure Platform
---
<!-- NOTE: This file was partially generated using AI assistance.  -->
# Running an experiment from a branch on the Every Cure Platform

This guide explains how to run an experiment from a specific branch using the Every Cure Platform's pipeline submission tool.

## Prerequisites

Before you begin, ensure you have the following:

1. Access to the Every Cure Platform
2. `gcloud` CLI installed and configured / authenticated
3. `kubectl` CLI installed (will be installed automatically if not present)
4. Docker installed (for building and pushing images)

Configure Docker to use the Google Container Registry:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

## Submitting a Pipeline Run

To submit a pipeline run, use the `kedro submit` command. This command builds a Docker image, creates an Argo workflow template, and submits the workflow to the Kubernetes cluster.

### Basic Usage

```bash
kedro submit --username <your-name>
```

This command will:
1. Check and set up dependencies (gcloud, kubectl)
2. Build and push a Docker image with your username as the tag
3. Generate an Argo workflow template
4. Create a namespace if it doesn't exist
5. Apply the Argo workflow template
6. Submit the workflow to Argo

### Command Options

- `--username`: (Required) Your username for tagging the Docker image and creating the namespace.
- `--namespace`: (Optional) Specify a custom namespace. Default is `dev-<username>`.
- `--run-name`: (Optional) Specify a custom run name. If not provided, it will be generated based on your current Git branch.
- `--verbose`: (Optional) Enable verbose output for debugging.

### Examples

1. Basic submission:
   ```bash
   kedro submit --username johndoe
   ```

2. Custom namespace and run name:
   ```bash
   kedro submit --username johndoe --namespace custom-namespace --run-name my-experiment
   ```

3. Verbose output:
   ```bash
   kedro submit --username johndoe --verbose
   ```

## Monitoring Your Workflow

After submitting the workflow, you'll be provided with instructions on how to monitor its progress:

1. To watch the workflow progress in the terminal:
   ```bash
   argo watch -n <namespace> <job-name>
   ```

2. To view the workflow in the Argo UI:
   ```bash
   argo get -n <namespace> <job-name>
   ```

3. You'll also be prompted to open the workflow in your browser. If you choose to do so, it will open the Argo UI for your specific workflow.

## Understanding the Pipeline

For a detailed overview of the pipeline stages (Preprocessing, Ingestion, Integration, Embeddings, Modelling, Evaluation, and Release), please refer to the [Pipeline documentation](../../onboarding/pipeline.md)

## Environments

The Every Cure Platform supports multiple environments. When submitting a pipeline run,
it will use the `cloud` environment by default, which is configured to read and write
data from/to GCP resources. For more information on available environments and their
configurations, see the [Pipeline documentation](../../onboarding/pipeline.md#environments).

## Troubleshooting

If you encounter any issues during the submission process:

1. Check the error messages in the console output.
2. Ensure all prerequisites are met and properly configured.
3. Use the `--verbose` flag for more detailed output.
4. If the issue persists, contact the platform support team with the error details and your run configuration.

Remember to update your [onboarding issue](https://github.com/everycure-org/matrix/issues/new?assignees=&labels=onboarding&projects=&template=onboarding.md&title=%3Cfirstname%3E+%3Clastname%3E) if you encounter any problems or have suggestions for improving this process.
```
