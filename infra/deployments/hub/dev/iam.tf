locals {
  matrix_all_group = "group:matrix-all@everycure.org"

  matrix_viewers_group = [local.matrix_all_group, "group:mtrx-viewers@everycure.org"]

  data_team_members = [
    "user:alexei@everycure.org",
  ]

  platform_team_members = [
    "user:pascal@everycure.org",
    "user:laurens@everycure.org"
  ]
}

module "project_iam_bindings" {
  source  = "terraform-google-modules/iam/google//modules/projects_iam"
  version = "~> 8.0"

  projects = [var.project_id]
  mode     = "additive"

  bindings = {
    # Editor permissions
    "roles/editor" = local.platform_team_members

    # Artifact Registry access
    "roles/artifactregistry.writer" = concat(
      local.data_team_members,
      local.platform_team_members
    )

    # GCS object creator
    "roles/storage.objectCreator" = ["group:techteam@everycure.org"]

    # Viewer permissions
    "roles/viewer" = local.matrix_viewers_group
    "roles/bigquery.jobUser" = local.matrix_viewers_group
    "roles/bigquery.dataViewer" = local.matrix_viewers_group
    "roles/bigquery.studioUser" = local.matrix_viewers_group
    "roles/bigquery.user" = local.matrix_viewers_group  
    "roles/iap.httpsResourceAccessor" = local.matrix_viewers_group

    # All Matrix members permissions
    "roles/compute.networkUser" = [local.matrix_all_group]
  }
}