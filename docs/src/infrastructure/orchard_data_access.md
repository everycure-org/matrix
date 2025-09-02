# Orchard Data Access Integration

## Overview

The Matrix platform (Production environment) has been configured to access orchard datasets from both development and production orchard environments. This enables Matrix Production users to query and analyze orchard data as part of their workflows.

## Implementation Details

### Custom IAM Roles

Two custom IAM roles have been created to provide secure, read-only access to orchard BigQuery datasets:

- **`bigQueryReadFromOrchardDev`**: Grants access to orchard development datasets
- **`bigQueryReadFromOrchardProd`**: Grants access to orchard production datasets

#### Permissions Granted

Each custom role includes the following permissions:
- `bigquery.datasets.get` - Retrieve dataset metadata
- `bigquery.tables.get` - Retrieve table metadata
- `bigquery.tables.list` - List tables within datasets
- `bigquery.tables.getData` - Read table data
- `bigquery.jobs.create` - Create BigQuery jobs for queries
- `bigquery.readsessions.create` - Create read sessions for optimal performance

### Service Account Access

The following service accounts have been granted access to orchard datasets:

#### Matrix User Service Accounts
- `sa-internal-data-science@mtrx-hub-prod-sms.iam.gserviceaccount.com` - For internal data science team
- `sa-subcon-standard@mtrx-hub-prod-sms.iam.gserviceaccount.com` - For standard contractors
- `sa-subcon-embiology@mtrx-hub-prod-sms.iam.gserviceaccount.com` - For embiology contractors

#### Kubernetes Service Account
- `sa-k8s-node@mtrx-hub-prod-sms.iam.gserviceaccount.com` - For Kubernetes workloads running in the Matrix cluster to access Orchard Data.

### Cross-Project Access

The IAM bindings are configured to grant Matrix service accounts access to the external orchard projects:
- **Development**: `ec-orchard-dev`
- **Production**: `ec-orchard-prod`

## Usage in Matrix Pipelines

### Kedro Dataset Configuration

Orchard datasets can be accessed through the Matrix pipeline using the existing BigQuery dataset configuration in the catalog:

```yaml
orchard_edges:
  type: import matrix_gcp_datasets.gcp.SparkBigQueryDataset
  project: ec-orchard-prod
  dataset: orchard_us
  table: latest_status
  shard: ${globals:data_sources.orchard.version}
```

### Access from Kubernetes Workloads

When running in the Matrix Kubernetes cluster, workloads automatically inherit the necessary permissions through the cluster's node service account. No additional configuration is required for basic access.

### Local Development Access

For local development, users should:

1. Ensure they are members of the appropriate Google Groups:
   - `data-science@everycure.org` (for internal team)
   - `ext.subcontractors.standard@everycure.org` (for standard contractors)
   - `ext.subcontractors.embiology@everycure.org` (for embiology contractors)

2. Set up service account impersonation in their local environment:
   ```bash
   export SPARK_IMPERSONATION_SERVICE_ACCOUNT=sa-internal-data-science@mtrx-hub-prod-sms.iam.gserviceaccount.com
   ```

## Security Considerations

- **Read-Only Access**: All permissions are read-only, preventing accidental modification of orchard data
- **Least Privilege**: Only the minimum required permissions are granted
- **Cross-Project Isolation**: Orchard data remains in separate GCP projects with controlled access
- **Audit Logging**: All access is logged through GCP audit logs for security monitoring

## Monitoring and Troubleshooting

### Common Issues

1. **Permission Denied Errors**: Verify that the service account has been granted the appropriate orchard access roles
2. **Project Not Found**: Ensure the orchard project IDs are correct in the configuration
3. **Quota Limits**: Monitor BigQuery usage to avoid exceeding orchard project quotas

### Debugging Access

To debug access issues:

1. Check IAM bindings in the orchard projects
2. Verify service account impersonation is configured correctly
3. Review audit logs for access attempts

## Infrastructure Code

The orchard access configuration is managed through Terraform in the following files:
- `infra/deployments/hub/prod/custom_role.tf` - Custom IAM role definitions
- `infra/deployments/hub/prod/iam.tf` - IAM bindings and access grants
- `infra/deployments/hub/prod/locals.tf` - Project IDs and service account definitions
