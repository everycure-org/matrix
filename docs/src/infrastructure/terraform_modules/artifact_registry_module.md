# Artifact Registry Module

The Artifact Registry module is a reusable Terraform module designed to create and manage Google Cloud Artifact Registry repositories with automated cleanup policies. This module is specifically configured to help manage container images with cost-effective retention policies.

## Overview

This module creates a Google Cloud Artifact Registry repository with flexible cleanup policies that automatically delete old artifacts while optionally keeping a minimum number of recent versions. The default configuration is optimized for container image management with a 3-day retention policy.

**Recent Updates**: The module has been simplified to focus on core cleanup functionality. The specialized `sample-run` image cleanup policies have been replaced (as it was buggy) with a more robust Argo Workflow-based cleanup system that automatically deletes Docker images after successful workflow completion. Only Failed Workflow images will be deleted after 14 days, whereas, successful ones will be deleted immediately through the Argo Workflow clean up task.

## Features

- **Automated Cleanup**: Automatically delete artifacts older than a specified number of days
- **Active Cleanup Policies**: Cleanup policies are enabled by default (`cleanup_policy_dry_run = false`)
- **Workflow-Based Image Deletion**: Complements Argo Workflows for automatic image cleanup after successful workflow completion.
- **Version Retention**: Optionally keep a minimum number of recent versions regardless of age
- **Flexible Configuration**: Support for different artifact formats (Docker, Maven, NPM, etc.)
- **Cost Optimization**: Default 3-day retention policy helps minimize storage costs
- **Simplified Policy Management**: Focused on failed workflow image cleanup to enable retry within the defined period.

## Usage

### Basic Usage

```terraform
module "artifact_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id    = "your-gcp-project"
  location      = "us-central1"
  repository_id = "my-docker-repo"
  description   = "Container images for my application"
}
```

### Advanced Configuration

```terraform
module "artifact_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id              = "your-gcp-project"
  location                = "us-central1"
  repository_id           = "my-docker-repo"
  format                  = "DOCKER"
  description             = "Production container images"
  delete_older_than_days  = "14d"     # Keep images for 14 days instead of default 3
  keep_count              = 5        # Always keep the 5 most recent versions
}
```

### Disable Cleanup Policies

To disable age-based deletion, set `delete_older_than_days` to null:

```terraform
module "artifact_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id              = "your-gcp-project"
  location                = "us-central1"
  repository_id           = "long-term-storage"
  delete_older_than_days  = null     # Disable age-based deletion
  keep_count              = 0        # Disable count-based retention
}
```

## Input Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_id` | `string` | - | **Required.** The GCP project ID where the repository will be created |
| `location` | `string` | - | **Required.** The location for the Artifact Registry repository (e.g., `us-central1`) |
| `repository_id` | `string` | - | **Required.** The ID of the repository (must be unique within the project and location). Must be 5-63 characters, start with a lowercase letter, followed by lowercase letters, digits, or dashes |
| `format` | `string` | `"DOCKER"` | The format of the repository. Supported values: `DOCKER`, `MAVEN`, `NPM` |
| `description` | `string` | `"Managed by Terraform."` | A description for the repository |
| `delete_older_than_days` | `string` | `"3d"` | Duration after which to delete images (e.g., `"3d"`, `"14d"`). Set to `null` to disable age-based deletion. Must be in format `"<number>d"` |
| `keep_count` | `number` | `0` | Minimum number of recent versions to keep regardless of age. Set to `0` to disable |

## Outputs

| Output | Description |
|--------|-------------|
| `repository_name` | The full name of the created repository |
| `repository_url` | The URL of the repository for pushing/pulling artifacts |

## Cleanup Policies

The module implements automated cleanup policies that work together with Argo Workflow-based image deletion:

### 1. Age-Based Deletion (DELETE Policy)

- **Purpose**: Automatically delete artifacts older than a specified duration. This is introduced so that failed workflow could be retried/restarted before deleting the images forever.
- **Default**: 14 days (`"14d"`) - cost-optimized for development environments
- **Format**: Must be specified as a string with 'd' suffix (e.g., `"1d"`, `"7d"`, `"30d"`)
- **Condition**: Applies to all artifacts older than the specified threshold
- **Disable**: Set `delete_older_than_days = null`
- **Active by Default**: Cleanup policies are enabled (`cleanup_policy_dry_run = false`)

### 2. Version Retention (KEEP Policy)

- **Purpose**: Preserve a minimum number of recent versions regardless of age
- **Default**: Disabled (`keep_count = 0`)
- **Condition**: Keeps the most recent versions based on the specified count
- **Use Case**: Ensure critical versions are always available for rollbacks

## Examples

### Development Environment

For development environments where images change frequently and storage costs should be minimized:

```terraform
module "dev_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id              = "myproject-dev"
  location                = "us-central1"
  repository_id           = "dev-images"
  description             = "Development container images"
  delete_older_than_days  = "1d"     # Very aggressive cleanup
  keep_count              = 3        # But always keep 3 recent versions
}
```

### Production Environment

For production environments where stability and rollback capability are important:

```terraform
module "prod_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id              = "myproject-prod"
  location                = "us-central1"
  repository_id           = "prod-images"
  description             = "Production container images"
  delete_older_than_days  = "30d"    # Keep for 30 days
  keep_count              = 10       # Always keep 10 recent versions
}
```

## Best Practices

1. **Environment-Specific Configuration**: Use different retention policies for different environments:
   - **Development**: Aggressive cleanup (1-3 days) to minimize costs
   - **Staging**: Moderate retention (7-14 days) for testing cycles
   - **Production**: Conservative retention (30+ days) with version keeping

2. **Duration Format**: Always specify durations with the "d" suffix (e.g., `"3d"`, `"14d"`, `"30d"`).

3. **Workflow Integration**: The module now automatically integrates with Argo Workflows for immediate cleanup after successful runs, reducing dependency on scheduled policies alone.

4. **Monitoring**: Monitor storage costs and adjust retention policies based on actual usage patterns.

5. **Version Retention**: Consider setting `keep_count > 0` for production repositories to ensure rollback capability.

6. **Format Selection**: Choose the appropriate format for your artifacts:
   - `DOCKER` for container images

7. **Cleanup Policy Status**: Cleanup policies are active by default. Monitor the Google Cloud Console to verify policies are working as expected.

## Integration with CI/CD

This module is designed to work seamlessly with CI/CD pipelines. After creating the repository, use the `repository_url` output to configure your build systems:

```terraform
# In your CI/CD configuration
output "docker_registry_url" {
  value = module.artifact_registry.repository_url
}
```

Then in your GitHub Actions or other CI systems:
```yaml
- name: Push to Artifact Registry
  run: |
    docker tag my-app:latest ${{ env.REGISTRY_URL }}/my-app:latest
    docker push ${{ env.REGISTRY_URL }}/my-app:latest
```

### Prerequisites

For workflow-based cleanup to work, ensure:

```terraform
# Example: Compute cluster with artifact registry and proper IAM
module "compute_cluster" {
  source = "../../../modules/stacks/compute_cluster"
  
  # ... other configuration
  
  # The compute cluster automatically creates the image repository
  # and configures IAM with artifactregistry.admin role
}
```

### Benefits

- **Immediate cleanup**: Images deleted right after successful workflow completion
- **Cost effective**: Reduces storage duration compared to scheduled-only cleanup
- **Selective**: Only deletes the specific image used by the completed workflow
- **Secure**: Uses existing infrastructure without additional service accounts
- **Robust**: Graceful error handling doesn't impact workflow success

For detailed implementation information about the workflow-based cleanup system, see the [Docker Cleanup Implementation Guide](../DOCKER_CLEANUP_IMPLEMENTATION.md).

## Cost Optimization

The updated module provides multiple layers of cost optimization:

### Scheduled Cleanup
- **Default 14-day retention**: Container images are automatically deleted after 14 days
- **Configurable retention**: Easily adjust retention periods based on environment needs
- **Active policies**: Cleanup policies are enabled by default (`cleanup_policy_dry_run = false`)

### Workflow-Based Cleanup  
- **Immediate deletion**: Docker images are deleted immediately after successful Argo Workflow completion
- **Selective cleanup**: Only deletes the specific image used by the workflow if failed

### Cost Benefits
- **Reduced storage costs**: Combination of immediate and scheduled cleanup minimizes image storage duration.
- **Development velocity**: 14-day retention policy supports typical development and testing cycles.
- **Production flexibility**: Easily override defaults for production environments requiring longer retention.
- **Retry Compatibility**: Allows for Workflows to be retried if failed, so that we do not need to deploy the image again.

## Troubleshooting

### Common Issues

1. **Cleanup Not Working**: 
   - Ensure cleanup policies are enabled (`cleanup_policy_dry_run = false`)
   - Verify the service account has `artifactregistry.admin` permissions
   - Check that the `delete_older_than_days` format includes the "d" suffix

2. **Invalid Duration Format**: 
   - Use string format with "d" suffix: `"3d"`, `"7d"`, `"30d"`
   - Avoid numeric values without the suffix

3. **Workflow Cleanup Failures**:
   - Verify GKE node service account has `roles/artifactregistry.admin` role
   - Check Argo Workflow logs for image deletion step details
   - Ensure workflow completion triggers the exit handler

4. **Repository Already Exists**: 
   - Repository ID must be unique within the project and location
   - Choose a different `repository_id` value

5. **Repository ID Validation Error**:
   - Repository ID must be 5-63 characters long
   - Must start with a lowercase letter
   - Can contain lowercase letters, digits, and dashes only

### Viewing Cleanup Policies

You can view the applied cleanup policies in the Google Cloud Console:
1. Navigate to Artifact Registry
2. Select your repository  
3. Go to the "Cleanup" tab
4. Verify policies show as "Active" not "Dry run"

### Workflow Cleanup Monitoring

Monitor workflow-based cleanup:
1. Check Argo Workflow logs for cleanup handler execution
2. Verify image deletion success/failure messages
3. Confirm images are removed from Artifact Registry after successful workflows

## Security Considerations

- Ensure proper IAM permissions are configured for repository access
- Consider using private repositories for sensitive artifacts
- Implement proper authentication for CI/CD systems accessing the registry

---

For questions or issues with this module, please refer to the Infrastructure team or create an issue in the repository.
