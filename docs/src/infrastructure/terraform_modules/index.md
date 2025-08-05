# Terraform Infrastructure Modules

This document provides an overview of all available Terraform modules in the Every Cure infrastructure. These modules are designed to be reusable across different environments and projects.

## Module Categories

Our modules are organized into three categories:

### Individual Components (`/modules/`)

These are standalone modules for specific infrastructure components:

#### [Artifact Registry](artifact_registry_module.md)
**Path**: `modules/artifact_registry`

Creates Google Cloud Artifact Registry repositories with automated cleanup policies.

**Key Features**:
- Automatic deletion of old artifacts (default: 3 days)
- Configurable retention policies for different environments
- Support for multiple artifact formats (Docker, Maven, NPM, etc.)
- Cost optimization through intelligent cleanup

**Usage**:
```terraform
module "my_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id    = "your-project-id"
  location      = "us-central1"
  repository_id = "my-app-images"
}
```

### Component Modules (`/modules/components/`)

These modules provide specific infrastructure components:

#### CloudBuild
**Path**: `modules/components/cloudbuild`

Sets up Google Cloud Build for CI/CD automation.

#### DNS
**Path**: `modules/components/dns`

Manages DNS zones and records.

#### Gateway
**Path**: `modules/components/gateway`

Configures ingress gateways and load balancers.

### Stack Modules (`/modules/stacks/`)

These are higher-level modules that combine multiple components:

#### Compute Cluster
**Path**: `modules/stacks/compute_cluster`

Creates a complete Kubernetes cluster with associated networking, storage, and security configurations. This module uses the Artifact Registry module internally.

#### Data Release Zone
**Path**: `modules/stacks/data_release_zone`

Sets up infrastructure for public data releases including CDN, storage, and access controls.

## Usage Guidelines

### Module Selection

1. **Use Individual Components** when you need a specific piece of infrastructure
2. **Use Component Modules** for common infrastructure patterns
3. **Use Stack Modules** for complete environment setups

### Best Practices

1. **Version Pinning**: Always specify module versions in production
2. **Environment Variables**: Use different configurations for dev/staging/prod
3. **Documentation**: Document any custom configurations or deviations
4. **Testing**: Test modules in development environments first

### Example: Setting Up a New Environment

```terraform
# Complete cluster setup
module "cluster" {
  source = "../../../modules/stacks/compute_cluster"
  # ... configuration
}

# Additional artifact registry for specific needs
module "special_registry" {
  source = "../../../modules/artifact_registry"
  
  project_id              = var.project_id
  location                = var.region
  repository_id           = "special-images"
  delete_older_than_days  = 7
  keep_count              = 5
}

# DNS for the environment
module "dns" {
  source = "../../../modules/components/dns"
  # ... configuration
}
```

## Contributing to Modules

When creating or modifying modules:

1. **Follow naming conventions**: Use descriptive, consistent names
2. **Include documentation**: Add README.md and variable descriptions
3. **Provide examples**: Include usage examples for common scenarios
4. **Test thoroughly**: Validate in multiple environments
5. **Update this document**: Add new modules to the appropriate section

## Module Development Standards

### Required Files

Every module should include:
- `main.tf` - Primary resource definitions
- `variables.tf` - Input variables with descriptions
- `outputs.tf` - Output values
- `README.md` - Usage documentation and examples

### Optional Files

- `versions.tf` - Provider version constraints
- `locals.tf` - Local value definitions
- `data.tf` - Data source definitions

### Documentation Requirements

1. **Variable descriptions**: All variables must have clear descriptions
2. **Output descriptions**: All outputs must be documented
3. **Usage examples**: Include at least one basic example
4. **Dependencies**: Document any external dependencies

## Getting Help

- **Module-specific questions**: Check the module's README.md first
- **General infrastructure questions**: Refer to the main [infrastructure documentation](index.md)
- **Issues or bugs**: Create an issue in the repository
- **New module requests**: Discuss with the Infrastructure team

---

*This documentation is maintained by the Infrastructure team. Last updated: July 2025*
