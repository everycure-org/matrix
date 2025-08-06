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

# Binding for standard contractors (excludes embiology)
resource "google_storage_bucket_iam_member" "object_user_standard" {
  bucket = var.storage_bucket_name
  role   = google_project_iam_custom_role.read_and_no_delete_or_overwrite_storage_role.id

  member = resource.google_service_account.sa["external_subcon_standard"].member

  condition {
    title       = "exclude_embiology_for_standard"
    description = "Standard contractors cannot access embiology folder."
    expression  = local.path_or_exclude_embiology
  }
}

# Binding for embiology contractors and internal team (includes embiology)
resource "google_storage_bucket_iam_member" "object_user_embiology_and_internal" {
  for_each = toset([
    resource.google_service_account.sa["external_subcon_embiology"].member,
    resource.google_service_account.sa["internal_data_science"].member
  ])
  bucket = var.storage_bucket_name
  role   = google_project_iam_custom_role.read_and_no_delete_or_overwrite_storage_role.id

  member = each.value

  condition {
    title       = "include_embiology_and_others"
    description = "Allows access to embiology and other allowed paths."
    expression = join(" || ", [
      for p in concat(local.allowed_paths, [local.embiology_path]) :
      "resource.name.startsWith(\"${p}\")"
    ])
  }
}

# Unconditional bucket listing permissions for embiology and internal team
# This is required because IAM conditions on storage.objects.list prevent bucket-level listing
resource "google_storage_bucket_iam_member" "bucket_list_embiology_and_internal" {
  for_each = toset([
    resource.google_service_account.sa["external_subcon_embiology"].member,
    resource.google_service_account.sa["internal_data_science"].member,
    resource.google_service_account.sa["external_subcon_standard"].member
  ])
  bucket = var.storage_bucket_name
  role   = "roles/storage.legacyBucketReader"
  member = each.value
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

