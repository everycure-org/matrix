locals {
  embiology_path_raw = "data/01_RAW/embiology"
  dev_bucket_name    = "mtrx-us-central1-hub-dev-storage"

  allowed_paths = [
    "projects/_/buckets/${var.storage_bucket_name}/objects/data/",
    "projects/_/buckets/${var.storage_bucket_name}/objects/data_releases/",
    "projects/_/buckets/${var.storage_bucket_name}/objects/kedro/data/",
  ]

  embiology_path = "projects/_/buckets/${var.storage_bucket_name}/objects/${local.embiology_path_raw}/"

  # Helper to build a logical OR expression for paths
  path_or_expression = join(" || ", [
    for p in local.allowed_paths : "resource.name.startsWith(\"${p}\")"
  ])

  # Same as above, but excludes embiology
  path_or_exclude_embiology = "( ${local.path_or_expression} ) && !resource.name.startsWith(\"${local.embiology_path}\")"

  # All allowed paths including embiology
  path_or_include_embiology = join(" || ", concat(local.allowed_paths, [local.embiology_path]))

  # Flatten the members for IAM bindings
  binding_members = [
    "serviceAccount:${resource.google_service_account.sa["external_subcon_standard"].email}",
    "serviceAccount:${resource.google_service_account.sa["internal_data_science"].email}",
    "serviceAccount:${resource.google_service_account.sa["external_subcon_embiology"].email}"
  ]
}

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

resource "google_project_iam_custom_role" "custom_storage_role" {
  project     = var.project_id
  role_id     = "customStorageAccess"
  title       = "Custom Storage Access"
  description = "Custom role with fine-grained storage and metadata permissions"

  permissions = [
    "storage.folders.create",
    "storage.folders.list",
    "storage.folders.get",
    "storage.managedFolders.create",
    "storage.managedFolders.list",
    "storage.managedFolders.get",
    "storage.multipartUploads.create",
    "storage.multipartUploads.abort",
    "storage.multipartUploads.listParts",
    "storage.multipartUploads.list",
    "storage.objects.create",
    "storage.objects.get",
    "storage.objects.list",
    "storage.buckets.get",
    "storage.objects.list",
    "storage.managedFolders.get",
    "storage.managedFolders.list",
    "storage.multipartUploads.list",
  ]
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

resource "google_project_iam_custom_role" "bigquery_read_write_no_delete" {
  role_id     = "bigQueryReadWriteNoDelete"
  title       = "BigQuery Read/Write Without Delete"
  description = "Allows read and insert access to BigQuery tables without delete/overwrite."
  project     = var.project_id

  permissions = [
    "bigquery.datasets.get",
    "bigquery.tables.get",
    "bigquery.tables.list",
    "bigquery.tables.create",
    "bigquery.tables.updateData",
    "bigquery.jobs.create",
    "bigquery.readsessions.create",
    "bigquery.tables.getData"
  ]
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
