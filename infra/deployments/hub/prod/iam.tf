# This module sets up IAM roles and permissions for various service accounts.
module "project_iam_bindings" {
  source   = "terraform-google-modules/iam/google//modules/projects_iam"
  projects = [var.project_id]
  version  = "~> 8.0"

  mode = "additive"

  bindings = {
    "roles/container.clusterViewer" = local.binding_members
    "roles/artifactregistry.writer" = local.binding_members
    "roles/container.developer"     = local.binding_members
  }
  depends_on = [google_service_account.sa]
}



resource "google_project_iam_member" "bigquery_read_write_no_delete" {
  for_each = toset(local.binding_members)
  project  = var.project_id
  role     = google_project_iam_custom_role.bigquery_read_write_no_delete.id
  member   = each.value
}

// Granting read access to the dev bucket for prod service accounts.
resource "google_storage_bucket_iam_member" "dev_bucket_read_access_for_prod" {
  for_each = toset(local.binding_members)
  bucket   = local.dev_bucket_name
  role     = "roles/storage.objectViewer"
  member   = each.value
}

// Granting read access to the dev bucket for prod service accounts.
resource "google_storage_bucket_iam_member" "github_actions_rw_dev_bucket_access_to_prod_read_only" {
  for_each = toset(local.binding_members)
  bucket   = var.storage_bucket_name
  role     = "roles/storage.objectViewer"
  member   = local.matrix_hub_dev_github_sa_rw_member
}