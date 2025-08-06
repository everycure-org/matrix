# Artifact Registry Module

The Artifact Registry module is a reusable Terraform module designed to create and manage Google Cloud Artifact Registry repositories with automated cleanup policies. This module is specifically configured to help manage container images with cost-effective retention policies.

## Overview

This module creates a Google Cloud Artifact Registry repository with flexible cleanup policies that automatically delete old artifacts while optionally keeping a minimum number of recent versions. The default configuration is optimized for container image management with a 3-day retention policy.

## Features

- **Automated Cleanup**: Automatically delete artifacts older than a specified number of days
- **Version Retention**: Optionally keep a minimum number of recent versions regardless of age
- **Flexible Configuration**: Support for different artifact formats (Docker, Maven, NPM, etc.)
- **Cost Optimization**: Default 3-day retention policy helps minimize storage costs
- **Tag State Management**: Configurable tag state filtering for cleanup policies

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
  delete_older_than_days  = 7      # Keep images for 7 days instead of default 3
  keep_count              = 5      # Always keep the 5 most recent versions
  tag_state               = "TAGGED"  # Only apply policies to tagged images
}
```

### Disable Cleanup Policies

```terraform
module "artifact_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id              = "your-gcp-project"
  location                = "us-central1"
  repository_id           = "long-term-storage"
  delete_older_than_days  = 0      # Disable age-based deletion
  keep_count              = 0      # Disable count-based retention
}
```

## Input Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_id` | `string` | - | **Required.** The GCP project ID where the repository will be created |
| `location` | `string` | - | **Required.** The location for the Artifact Registry repository (e.g., `us-central1`) |
| `repository_id` | `string` | - | **Required.** The ID of the repository (must be unique within the project and location) |
| `format` | `string` | `"DOCKER"` | The format of the repository. Supported values: `DOCKER`, `MAVEN`, `NPM`, `PYTHON`, `APT`, `YUM` |
| `description` | `string` | `"Managed by Terraform."` | A description for the repository |
| `delete_older_than_days` | `number` | `3` | Number of days after which to delete images. Set to `0` to disable age-based deletion |
| `keep_count` | `number` | `0` | Minimum number of recent versions to keep regardless of age. Set to `0` to disable |
| `tag_state` | `string` | `"ANY"` | Tag state to apply cleanup policies to. Options: `ANY`, `TAGGED`, `UNTAGGED` |

## Outputs

| Output | Description |
|--------|-------------|
| `repository_name` | The full name of the created repository |
| `repository_url` | The URL of the repository for pushing/pulling artifacts |

## Cleanup Policies

The module implements two types of cleanup policies that work together:

### 1. Age-Based Deletion (DELETE Policy)

- **Purpose**: Automatically delete artifacts older than a specified number of days
- **Default**: 3 days (cost-optimized for development environments)
- **Condition**: Applies to all artifacts older than the specified threshold
- **Disable**: Set `delete_older_than_days = 0`

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
  delete_older_than_days  = 1      # Very aggressive cleanup
  keep_count              = 3      # But always keep 3 recent versions
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
  delete_older_than_days  = 30     # Keep for 30 days
  keep_count              = 10     # Always keep 10 recent versions
  tag_state               = "TAGGED"  # Only manage tagged releases
}
```

## Best Practices

1. **Environment-Specific Configuration**: Use different retention policies for different environments:
   - **Development**: Aggressive cleanup (1-3 days) to minimize costs
   - **Staging**: Moderate retention (7-14 days) for testing cycles
   - **Production**: Conservative retention (30+ days) with version keeping

2. **Tag Management**: Use meaningful tags and consider setting `tag_state = "TAGGED"` for production repositories to only manage properly tagged releases.

3. **Monitoring**: Monitor storage costs and adjust retention policies based on actual usage patterns.

4. **Version Retention**: Always set `keep_count > 0` for production repositories to ensure rollback capability.

5. **Format Selection**: Choose the appropriate format for your artifacts:
   - `DOCKER` for container images
   - `MAVEN` for Java libraries
   - `NPM` for Node.js packages
   - `PYTHON` for Python packages

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

## Cost Optimization

The default 3-day retention policy is specifically designed to balance functionality with cost optimization:

- **Storage Costs**: Container images can be large; keeping them for extended periods can be expensive
- **Development Velocity**: 3 days is typically sufficient for development and testing cycles
- **Override Available**: Production environments can easily override this default

## Troubleshooting

### Common Issues

1. **Cleanup Not Working**: Ensure the service account running Terraform has the `artifactregistry.repositories.update` permission.

2. **Images Not Being Deleted**: Check that your images have the correct tag state if you've configured `tag_state` to something other than `"ANY"`.

3. **Repository Already Exists**: The repository ID must be unique within the project and location. Choose a different `repository_id`.

### Viewing Cleanup Policies

You can view the applied cleanup policies in the Google Cloud Console:
1. Navigate to Artifact Registry
2. Select your repository
3. Go to the "Cleanup" tab

## Security Considerations

- Ensure proper IAM permissions are configured for repository access
- Consider using private repositories for sensitive artifacts
- Implement proper authentication for CI/CD systems accessing the registry

---

For questions or issues with this module, please refer to the Infrastructure team or create an issue in the repository.
