locals {
  matrix_all_group = "group:matrix-all@everycure.org"

  matrix_viewers_group = [local.matrix_all_group, "group:matrix-viewers@everycure.org"]
  tech_team_group      = ["group:techteam@everycure.org", "group:ext.tech.dataminded@everycure.org"]
}

module "project_iam_bindings" {
  source   = "terraform-google-modules/iam/google//modules/projects_iam"
  projects = [module.bootstrap_data.content.project_id]
  version  = "~> 8.0"

  mode = "additive"

  bindings = {
    "roles/bigquery.studioAdmin"           = local.tech_team_group
    "roles/notebooks.admin"                = local.tech_team_group
    "roles/ml.admin"                       = local.tech_team_group
    "roles/aiplatform.admin"               = local.tech_team_group
    "roles/ml.developer"                   = local.tech_team_group
    "roles/artifactregistry.writer"        = local.tech_team_group
    "roles/storage.objectCreator"          = local.tech_team_group
    "roles/storage.objectUser"             = local.tech_team_group
    "roles/container.clusterAdmin"         = local.tech_team_group
    "roles/container.developer"            = local.tech_team_group
    "roles/compute.admin"                  = local.tech_team_group
    "roles/iam.workloadIdentityPoolAdmin"  = local.tech_team_group
    "roles/iam.serviceAccountTokenCreator" = local.tech_team_group
    "roles/viewer"                         = local.matrix_viewers_group
    "roles/bigquery.jobUser"               = local.matrix_viewers_group
    "roles/bigquery.dataViewer"            = local.matrix_viewers_group
    "roles/bigquery.studioUser"            = local.matrix_viewers_group
    "roles/bigquery.user"                  = local.matrix_viewers_group
    "roles/iap.httpsResourceAccessor"      = local.matrix_viewers_group

    "roles/compute.networkUser" = [local.matrix_all_group]
  }

  conditional_bindings = [
    {
      role        = "roles/storage.objectCreator"
      title       = "matrix_raw_data_access"
      description = "Allow matrix-all group to create objects only in RAW data folder"
      expression  = "resource.name.startsWith(\"projects/_/buckets/mtrx-us-central1-hub-dev-storage/objects/data/01_RAW\")"
      members     = [local.matrix_all_group]
    }
  ]
}
