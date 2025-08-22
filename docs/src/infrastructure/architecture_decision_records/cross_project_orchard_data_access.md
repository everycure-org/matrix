# ADR: Cross-Project Orchard Data Access Implementation

**Status:** Accepted  
**Date:** 2025-01-25  
**Authors:** Infrastructure Team  
**Stakeholders:** Data Science Team, Pipeline Users  

## Context

The Matrix platform needed to access orchard datasets stored in separate GCP projects (`ec-orchard-dev` and `ec-orchard-prod`) to enable data scientists to include orchard data in their analysis and pipeline workflows. The orchard data is managed by a separate team and stored in their own GCP projects with their own access controls and billing.

## Decision

We implemented cross-project BigQuery access using custom IAM roles and service account impersonation, with the following key decisions:

### 1. Custom IAM Roles for Minimal Permissions

We created two custom IAM roles in the orchard projects:
- `bigQueryReadFromOrchardDev` - For development orchard data access  
- `bigQueryReadFromOrchardProd` - For production orchard data access

Each role includes only the minimum required permissions:
- `bigquery.datasets.get`
- `bigquery.tables.get` 
- `bigquery.tables.list`
- `bigquery.tables.getData`
- `bigquery.jobs.create`
- `bigquery.readsessions.create`

### 2. Service Account Access Pattern

We grant orchard access to multiple Matrix service accounts based on user roles:
- `sa-internal-data-science@mtrx-hub-prod-sms.iam.gserviceaccount.com` - Internal data science team
- `sa-subcon-standard@mtrx-hub-prod-sms.iam.gserviceaccount.com` - Standard contractors
- `sa-subcon-embiology@mtrx-hub-prod-sms.iam.gserviceaccount.com` - Embiology contractors
- `sa-k8s-node@mtrx-hub-prod-sms.iam.gserviceaccount.com` - Kubernetes workloads

### 3. Kubernetes Node Service Account Access

**Critical Decision:** In our GKE cluster, Workload Identity is disabled, meaning all pods inherit permissions from the Kubernetes node service account (`sa-k8s-node`) rather than individual pod service accounts. Therefore, we must grant orchard access permissions directly to the node service account for pipeline workloads to function.

### 4. Cross-Project IAM Binding Structure

The IAM bindings are applied in the **orchard projects** (not the Matrix project), granting Matrix service accounts access to orchard resources:

```hcl
# Example binding in orchard projects
resource "google_project_iam_member" "orchard_prod_access" {
  project = local.orchard_prod_project_id  # ec-orchard-prod
  role    = "projects/${local.orchard_prod_project_id}/roles/bigQueryReadFromOrchardProd"
  member  = "serviceAccount:${local.prod_k8s_sas.gke_node_sa}"
}
```

## Alternatives Considered

### 1. Shared Service Account
**Rejected:** Creating a single shared service account across projects would have created unnecessary coupling and security risks.

### 2. Direct Project-Level BigQuery Roles
**Rejected:** Using predefined BigQuery roles (like `roles/bigquery.dataViewer`) would grant excessive permissions beyond what's needed.

### 3. Workload Identity Implementation
**Considered but not implemented:** Enabling Workload Identity would allow pod-level service account isolation but would require significant infrastructure changes to our existing cluster setup.

### 4. Data Replication to Matrix Project
**Rejected:** Copying orchard data to the Matrix project would create data inconsistency issues and increase storage costs.

## Consequences

### Positive
- **Minimal Security Risk:** Custom roles provide least-privilege access to orchard data
- **Seamless Integration:** Users can access orchard data through standard Matrix pipeline patterns
- **Audit Trail:** All access is logged through GCP audit logs across both projects
- **Cost Efficiency:** No data replication or additional storage costs
- **Operational Simplicity:** Uses existing service account impersonation patterns

### Negative  
- **Cross-Project Complexity:** IAM management spans multiple GCP projects
- **Node SA Permissions:** All Kubernetes workloads inherit orchard access permissions (due to disabled Workload Identity)
- **Dependency Risk:** Changes to orchard project structure could impact Matrix access
- **Quota Coupling:** Heavy Matrix usage could impact orchard project BigQuery quotas

### Neutral
- **Documentation Requirement:** Requires clear documentation for users on how to access orchard data
- **Support Complexity:** Troubleshooting access issues requires understanding of cross-project setup

## Implementation Details

The implementation consists of three main Terraform files in `/infra/deployments/hub/prod/`:

1. **`custom_role.tf`** - Defines the custom IAM roles with minimal required permissions
2. **`iam.tf`** - Creates the cross-project IAM bindings  
3. **`locals.tf`** - Defines project IDs and service account references

## Monitoring and Maintenance

- **Access Monitoring:** GCP audit logs track all cross-project access attempts
- **Permission Reviews:** Custom roles should be reviewed quarterly to ensure minimal permissions
- **Dependency Tracking:** Monitor orchard project changes that could impact Matrix access
- **Cost Monitoring:** Track BigQuery usage in orchard projects attributable to Matrix workloads

## Future Considerations

1. **Workload Identity Migration:** Consider enabling Workload Identity in the future to provide pod-level service account isolation
2. **Automated Role Updates:** Implement automation to detect and adapt to changes in required BigQuery permissions
3. **Data Cataloging:** Consider implementing data cataloging tools to better discover and document available orchard datasets
4. **Quota Management:** Implement quota monitoring and alerting to prevent Matrix usage from impacting orchard operations

## References

- [Google Cloud Cross-Project IAM Documentation](https://cloud.google.com/iam/docs/cross-project-service-accounts)
- [BigQuery Custom Roles Best Practices](https://cloud.google.com/bigquery/docs/access-control)
- [GKE Workload Identity Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity)
- [Matrix Infrastructure Documentation](../index.md)
- [Orchard Data Access Guide](../orchard_data_access.md)
