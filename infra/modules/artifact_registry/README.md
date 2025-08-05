# Artifact Registry Terraform Module

A Terraform module for creating Google Cloud Artifact Registry repositories with automated cleanup policies.

## Quick Start

```terraform
module "my_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id    = "your-project-id"
  location      = "us-central1" 
  repository_id = "my-app-images"
}
```

This creates a Docker registry that automatically deletes images older than 3 days.

## Features

- 🔄 **Automatic cleanup** - Delete old artifacts to save storage costs
- 📦 **Multi-format support** - Docker, Maven, NPM, Python, and more
- 🛡️ **Version protection** - Keep minimum number of recent versions
- ⚙️ **Flexible configuration** - Customize retention policies per environment

## Variables

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|----------|
| project_id | GCP project ID | string | - | yes |
| location | Repository location | string | - | yes |
| repository_id | Repository ID | string | - | yes |
| format | Artifact format | string | "DOCKER" | no |
| description | Repository description | string | "Managed by Terraform." | no |
| delete_older_than_days | Days after which to delete artifacts | number | 3 | no |
| keep_count | Minimum versions to keep | number | 0 | no |
| tag_state | Tag state for cleanup policies | string | "ANY" | no |

## Outputs

| Name | Description |
|------|-------------|
| repository_name | Full repository name |
| repository_url | Repository URL for pushing/pulling |

## Examples

### Development Environment
```terraform
module "dev_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id              = "myproject-dev"
  location                = "us-central1"
  repository_id           = "dev-images"
  delete_older_than_days  = 1  # Aggressive cleanup
  keep_count              = 3  # But keep 3 recent versions
}
```

### Production Environment
```terraform
module "prod_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id              = "myproject-prod"
  location                = "us-central1"
  repository_id           = "prod-images"
  delete_older_than_days  = 30      # Keep longer
  keep_count              = 10      # Keep more versions
  tag_state               = "TAGGED" # Only tagged images
}
```

### Disable Cleanup
```terraform
module "archive_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id              = "myproject"
  location                = "us-central1"
  repository_id           = "archive"
  delete_older_than_days  = 0  # No age-based deletion
  keep_count              = 0  # No count-based retention
}
```

## Usage in CI/CD

After creating the registry, use the output URL in your CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Push to Artifact Registry
  run: |
    docker tag my-app:${{ github.sha }} ${{ env.REGISTRY_URL }}/my-app:${{ github.sha }}
    docker push ${{ env.REGISTRY_URL }}/my-app:${{ github.sha }}
```

## Cost Optimization

The default 3-day retention is designed to minimize storage costs while maintaining development velocity. Adjust based on your needs:

- **Development**: 1-3 days (cost optimization)
- **Staging**: 7-14 days (testing cycles) 
- **Production**: 30+ days (stability & rollbacks)

## Documentation

For detailed documentation, see: [docs/src/infrastructure/artifact_registry_module.md](../../../docs/src/infrastructure/artifact_registry_module.md)
