locals {
  internal_data_science     = "group:data-science@everycure.org"
  external_subcon_standard  = "group:ext.subcontractors@everycure.org"
  external_subcon_embiology = "group:ext.subcontractors@everycure.org"
  embiology_path_processed  = "data/01_RAW/KGs/embiology"
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
    "roles/bigquery.dataViewer"     = [local.internal_data_science]

  }

  conditional_bindings = [
    {
      role        = "roles/storage.objectCreator"
      title       = "matrix_raw_data_access"
      description = "Allow matrix-all group to create objects only in RAW data folder"
      expression  = "resource.name.startsWith(\"projects/_/buckets/mtrx-us-central1-hub-prod-storage/objects/data/01_RAW\")"
      members     = [local.matrix_all_group]
    },

    {
      role        = "roles/storage.objectViewer"
      title       = "external_subcons_gcs_except_embiology"
      description = "Allow standard contractors to view GCS objects, excluding the embiology folder."
      expression  = <<-EOT
        resource.name.startsWith("projects/_/buckets/${local.storage_bucket_name}/objects/") &&
        !resource.name.startsWith("projects/_/buckets/${local.storage_bucket_name}/objects/${local.embiology_path_processed}") &&
        !resource.name.startsWith("projects/_/buckets/${local.storage_bucket_name}/objects/${local.embiology_path_raw}")
      EOT
      members     = [local.external_subcon_standard]
    },
    {
      role        = "roles/storage.objectViewer"
      title       = "external_subcons_gcs_access_embiology"
      description = "Allow specific contractors to view GCS objects only in the embiology folder."
      expression  = <<-EOT
        resource.name.startsWith("projects/_/buckets/${local.storage_bucket_name}/objects/") &&
        (
          resource.name.startsWith("projects/_/buckets/${local.storage_bucket_name}/objects/${local.embiology_path_processed}") ||
          resource.name.startsWith("projects/_/buckets/${local.storage_bucket_name}/objects/${local.embiology_path_raw}")
        )
      EOT
      members     = [local.external_subcon_embiology]
    }
  ]
}
