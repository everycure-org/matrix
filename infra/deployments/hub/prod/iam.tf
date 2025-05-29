// This file contains IAM bindings for the production environment of the hub project.

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
}

# Binding for standard contractors (excludes embiology)
resource "google_storage_bucket_iam_binding" "object_user_standard" {
  bucket = var.storage_bucket_name
  role   = google_project_iam_custom_role.custom_storage_role.id

  members = [
    "serviceAccount:${resource.google_service_account.sa["external_subcon_standard"].email}"
  ]

  condition {
    title       = "exclude_embiology_for_standard"
    description = "Standard contractors cannot access embiology folder."
    expression  = local.path_or_exclude_embiology
  }
}

# Binding for embiology contractors and internal team (includes embiology)
resource "google_storage_bucket_iam_binding" "object_user_embiology_and_internal" {
  bucket = var.storage_bucket_name
  role   = google_project_iam_custom_role.custom_storage_role.id

  members = [
    "serviceAccount:${resource.google_service_account.sa["external_subcon_embiology"].email}",
    "serviceAccount:${resource.google_service_account.sa["internal_data_science"].email}"
  ]

  condition {
    title       = "include_embiology_and_others"
    description = "Allows access to embiology and other allowed paths."
    expression = join(" || ", [
      for p in concat(local.allowed_paths, [local.embiology_path]) :
      "resource.name.startsWith(\"${p}\")"
    ])
  }
}



resource "google_project_iam_member" "bigquery_read_write_no_delete" {
  for_each = toset(local.binding_members)
  project  = var.project_id
  role     = google_project_iam_custom_role.bigquery_read_write_no_delete.id
  member   = each.value
}

resource "google_storage_bucket_iam_binding" "dev_bucket_access" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.objectViewer"

  members = local.binding_members

  condition {
    title       = "access_dev_bucket"
    description = "Allows access to read dev only."
    expression  = "resource.name.startsWith(\"projects/_/buckets/mtrx-us-central1-hub-dev-storage/objects/data/01_RAW\")"
  }
}
