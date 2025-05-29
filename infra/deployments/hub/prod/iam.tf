locals {
  internal_data_science     = "group:data-science@everycure.org"
  external_subcon_standard  = "group:ext.subcontractors.standard@everycure.org"
  external_subcon_embiology = "group:ext.subcontractors.embiology@everycure.org"
  embiology_path_raw        = "data/01_RAW/embiology"

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
  binding_members = [local.internal_data_science, local.external_subcon_standard, local.external_subcon_embiology]
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

// Needed as due to the conditional expression, if any object (e.g., those in the embiology folder) is excluded by the condition, listing operations may be hindered.
// This binding allows external subcontractors to list all objects in the bucket.
resource "google_storage_bucket_iam_binding" "external_subcons_bucket_list" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.legacyBucketReader"
  members = [
    local.external_subcon_standard, local.internal_data_science, local.external_subcon_embiology
  ]
}

# Binding for standard contractors (excludes embiology)
resource "google_storage_bucket_iam_binding" "object_user_standard" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.objectCreator"

  members = [
    local.external_subcon_standard
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
  role   = "roles/storage.objectCreator"

  members = [
    local.external_subcon_embiology,
    local.internal_data_science
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

resource "google_project_iam_member" "assign_custom_bq_role" {
  for_each = toset(local.binding_members)
  project  = var.project_id
  role     = google_project_iam_custom_role.bigquery_read_write_no_delete.id
  member   = each.key
}
