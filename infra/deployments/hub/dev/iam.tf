locals {
  matrix_all_group                     = "group:matrix-all@everycure.org"
  prod_sas                             = ["serviceAccount:sa-k8s-node@mtrx-hub-prod-sms.iam.gserviceaccount.com"]
  matrix_viewers_group                 = [local.matrix_all_group, "group:matrix-viewers@everycure.org"]
  tech_team_group                      = ["group:techteam@everycure.org"]
  orchard_prod_compute_service_account = "serviceAccount:342224594736-compute@developer.gserviceaccount.com"
  orchard_dev_compute_service_account  = "serviceAccount:299386668624-compute@developer.gserviceaccount.com"
  cross_account_sas = [
    "serviceAccount:vertex-ai-workbench-sa@mtrx-wg1-data-dev-nb5.iam.gserviceaccount.com",
    "serviceAccount:vertex-ai-workbench-sa@mtrx-wg2-modeling-dev-9yj.iam.gserviceaccount.com",
    local.orchard_dev_compute_service_account, # orchard dev
    local.orchard_prod_compute_service_account # orchard prod
  ]
  github_actions_rw     = ["serviceAccount:sa-github-actions-rw@mtrx-hub-dev-3of.iam.gserviceaccount.com"]
  custom_cloud_build_sa = ["serviceAccount:custom-cloud-build-sa@mtrx-hub-prod-sms.iam.gserviceaccount.com"]
  arpah_read_only_group = ["group:arpah-read-only@everycure.org"]
}

module "project_iam_bindings" {
  source   = "terraform-google-modules/iam/google//modules/projects_iam"
  projects = [var.project_id]
  version  = "~> 8.0"

  mode = "additive"

  bindings = {
    "roles/bigquery.studioAdmin"           = local.tech_team_group
    "roles/notebooks.admin"                = local.tech_team_group
    "roles/ml.admin"                       = local.tech_team_group
    "roles/aiplatform.admin"               = local.tech_team_group
    "roles/ml.developer"                   = local.tech_team_group
    "roles/storage.objectCreator"          = local.tech_team_group
    "roles/container.clusterAdmin"         = local.tech_team_group
    "roles/container.developer"            = local.tech_team_group
    "roles/compute.admin"                  = local.tech_team_group
    "roles/iam.workloadIdentityPoolAdmin"  = local.tech_team_group
    "roles/iam.serviceAccountTokenCreator" = local.tech_team_group
    "roles/storage.objectUser"             = local.tech_team_group
    "roles/storage.objectViewer"           = flatten([local.cross_account_sas, local.arpah_read_only_group])
    "roles/storage.legacyBucketReader"     = local.arpah_read_only_group
    "roles/artifactregistry.writer"        = flatten([local.tech_team_group, [local.matrix_all_group]]) # enables people to run kedro experiment run
    "roles/viewer"                         = flatten([local.matrix_viewers_group, local.cross_account_sas, local.custom_cloud_build_sa])
    "roles/bigquery.jobUser"               = flatten([local.matrix_viewers_group, local.cross_account_sas])
    # giving prod k8s cluster access to our dev data. 
    "roles/bigquery.dataViewer"       = flatten([local.matrix_viewers_group, local.cross_account_sas, local.prod_sas])
    "roles/bigquery.studioUser"       = flatten([local.matrix_viewers_group, local.cross_account_sas])
    "roles/bigquery.user"             = flatten([local.matrix_viewers_group, local.cross_account_sas])
    "roles/iap.httpsResourceAccessor" = flatten([local.matrix_viewers_group, local.github_actions_rw])
    # Grant Cloud Build service account necessary permissions for storage operations
    "roles/storage.admin" = local.custom_cloud_build_sa
    # roles/owner does _not_ have org policy edit rights. the SA needs it though if we want to edit org policies (e.g. for overrides)
    # TODO: https://linear.app/everycure/issue/SEC-3
    # "roles/orgpolicy.policyAdmin" = local.custom_cloud_build_sa

    "roles/compute.networkUser" = [local.matrix_all_group]
  }
}
