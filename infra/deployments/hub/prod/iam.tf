locals {
  matrix_all_group          = "group:matrix-all@everycure.org"
  internal_data_science     = "group:data-science@everycure.org"
  external_subcon_standard  = "group:ext.subcontractors.standard@everycure.org"
  external_subcon_embiology = "group:ext.subcontractors.embiology@everycure.org"
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
      role        = "roles/storage.objectViewer"
      title       = "external_subcons_gcs_except_embiology"
      description = "Allow standard contractors to view GCS objects, excluding the embiology folder."
      expression  = <<-EOT
        resource.name.startsWith("projects/_/buckets/${var.storage_bucket_name}/objects/") &&
        !resource.name.startsWith("projects/_/buckets/${var.storage_bucket_name}/objects/${local.embiology_path_processed}") &&
        !resource.name.startsWith("projects/_/buckets/${var.storage_bucket_name}/objects/${local.embiology_path_raw}")
      EOT
      members     = [local.external_subcon_standard]
    },
    {
      role        = "roles/storage.objectViewer"
      title       = "individual_users_embiology_access"
      description = "Allow up to 10 specific contractors to view GCS objects only in the embiology folders."
      expression  = <<-EOT
        resource.name.startsWith("projects/_/buckets/${var.storage_bucket_name}/objects/") &&
        (
          resource.name.startsWith("projects/_/buckets/${var.storage_bucket_name}/objects/${local.embiology_path_processed}") ||
          resource.name.startsWith("projects/_/buckets/${var.storage_bucket_name}/objects/${local.embiology_path_raw}")
        )
      EOT
      members     = [local.external_subcon_embiology]
    }
  ]
}
