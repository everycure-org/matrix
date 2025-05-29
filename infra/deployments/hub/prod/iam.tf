locals {
  matrix_all_group          = "group:matrix-all@everycure.org"
  internal_data_science     = "group:data-science@everycure.org"
  external_subcon_standard  = "group:ext.subcontractors.standard@everycure.org"
  external_subcon_embiology = "group:ext.subcontractors.embiology@everycure.org"
  embiology_path_raw        = "data/01_RAW/embiology"
}

module "project_iam_bindings" {
  source   = "terraform-google-modules/iam/google//modules/projects_iam"
  projects = [var.project_id]
  version  = "~> 8.0"

  mode = "additive"

  bindings = {

    "roles/container.clusterViewer" = flatten([local.internal_data_science, local.external_subcon_standard])
    "roles/artifactregistry.writer" = flatten([local.internal_data_science, local.external_subcon_standard])
    "roles/storage.objectViewer"    = [local.internal_data_science]
    "roles/bigquery.dataViewer"     = flatten([local.internal_data_science, local.external_subcon_standard])
    "roles/container.developer"     = flatten([local.internal_data_science, local.external_subcon_standard])

  }

  conditional_bindings = [
    {
      role        = "roles/storage.objectCreator"
      title       = "matrix_raw_data_access"
      description = "Allow matrix-all group to create objects only in RAW data folder"
      expression  = "resource.name.startsWith(\"projects/_/buckets/${var.storage_bucket_name}/objects/data/01_RAW\")"
      members     = [local.matrix_all_group]
    },

    {
      role        = "roles/storage.objectUser"
      title       = "individual_users_embiology_access"
      description = "Allow up to 10 specific contractors to list, read, and write GCS objects, excluding the embiology folder."
      expression  = <<-EOT
        resource.name.startsWith("projects/_/buckets/${var.storage_bucket_name}/")
      EOT
      members     = [local.external_subcon_embiology]
    }
  ]
}

// Needed as due to the conditional expression, if any object (e.g., those in the embiology folder) is excluded by the condition, listing operations may be hindered.
// This binding allows external subcontractors to list all objects in the bucket.
resource "google_storage_bucket_iam_binding" "external_subcons_bucket_list" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.legacyBucketReader"
  members = [
    local.external_subcon_standard
  ]
}

resource "google_storage_bucket_iam_binding" "external_subcons_gcs_except_embiology" {
  bucket = var.storage_bucket_name

  role = "roles/storage.objectUser"

  members = [
    local.external_subcon_standard
  ]

  condition {
    title       = "external_subcons_gcs_except_embiology"
    description = "Allow standard contractors to list, read, and write GCS objects, excluding the embiology folder."
    expression  = <<-EOT
      (
      resource.name.startsWith("projects/_/buckets/${var.storage_bucket_name}/objects/data/") 
      ||
      resource.name.startsWith("projects/_/buckets/${var.storage_bucket_name}/objects/data_releases/") 
      ||
      resource.name.startsWith("projects/_/buckets/${var.storage_bucket_name}/objects/kedro/data/")
      )
      &&
      !resource.name.startsWith("projects/_/buckets/${var.storage_bucket_name}/objects/${local.embiology_path_raw}/")
    EOT
  }
}
