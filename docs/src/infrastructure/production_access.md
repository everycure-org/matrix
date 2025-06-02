# Production Access

This documentation explains the process of granting users production access through Google Cloud Identity Groups, impersonating service accounts (SA), and outlines the specific permissions each service account has.

## Step 1: Adding Users to Google Groups

To grant production access, users must be added to the appropriate Google Group corresponding to their role or team. To do this, please follow instructions at: https://support.google.com/groups/answer/2465464?hl=en

## Step 2: Service Accounts and Permissions

| **Role**                             | **Google Group**                                      | **Service Account**                                                           | **Permissions**                                                                                                                                                                 |
|--------------------------------------|--------------------------------------------------------|--------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Internal Data Science**            | data-science@everycure.org                             | sa-internal-data-science@mtrx-hub-prod-sms.iam.gserviceaccount.com            | - Access to embiology and other allowed paths  <br> - BigQuery read/write (no delete) <br> - Access to dev datasets <br> - View Kubernetes clusters (`roles/container.clusterViewer`) <br> - Write to Artifact Registry (`roles/artifactregistry.writer`) <br> - Manage containers (`roles/container.developer`) |
| **External Subcontractor (Embiology)** | ext.subcontractors.embiology@everycure.org             | sa-subcon-embiology@mtrx-hub-prod-sms.iam.gserviceaccount.com                 | - Access to embiology and other allowed paths <br> - BigQuery read/write (no delete) <br> - Access to dev datasets <br> - View Kubernetes clusters (`roles/container.clusterViewer`) <br> - Write to Artifact Registry (`roles/artifactregistry.writer`) <br> - Manage containers (`roles/container.developer`) |
| **External Subcontractor (Standard)** | ext.subcontractors.standard@everycure.org              | sa-subcon-standard@mtrx-hub-prod-sms.iam.gserviceaccount.com                  | - **Restricted from embiology folder** <br> - General access to permitted Cloud Storage paths <br> - BigQuery read/write (no delete) <br> - Access to dev datasets <br> - View Kubernetes clusters (`roles/container.clusterViewer`) <br> - Write to Artifact Registry (`roles/artifactregistry.writer`) <br> - Manage containers (`roles/container.developer`) |

## Step 3: Impersonating a Service Account

Once added to a group, users can impersonate a service account using the following gcloud CLI command:

```bash
gcloud auth login --update-adc --impersonate-service-account=<SERVICE_ACCOUNT_EMAIL>
```

Replace <SERVICE_ACCOUNT_EMAIL> with the service account email provided.


## Permission Details

### Cloud Storage Access

#### Internal and Embiology Contractors:
```bash
condition {
  title       = "include_embiology_and_others"
  description = "Allows access to embiology and other allowed paths."
  expression  = join(" || ", [
    for p in concat(local.allowed_paths, [local.embiology_path]) :
    "resource.name.startsWith(\"${p}\")"
  ])
}
```

#### Standard Contractors (Excluding Embiology):
```bash
condition {
  title       = "exclude_embiology_for_standard"
  description = "Standard contractors cannot access embiology folder."
  expression  = local.path_or_exclude_embiology
}
```


### BigQuery Permissions

#### Custom BigQuery role permissions

1. Read datasets and tables

2. Write to datasets and tables

3. Excludes delete/replace operations
```bash
resource "google_project_iam_member" "bigquery_read_write_no_delete" {
  for_each = toset(local.binding_members)
  project  = var.project_id
  role     = google_project_iam_custom_role.bigquery_read_write_no_delete.id
  member   = each.value
}
```

## Troubleshooting

**Permission Denied Errors:** Verify group membership and IAM permissions.

**Service Account Impersonation Issues:** Ensure IAM roles such as iam.serviceAccountUser and iam.serviceAccountTokenCreator are assigned correctly.