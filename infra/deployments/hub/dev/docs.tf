# IAM for documentation page. Only hosted in dev using AppEngine for now

# needed because our org policy disables the default permissions 
# https://stackoverflow.com/questions/43905748/what-permission-is-required-for-a-service-account-to-deploy-to-google-app-engine
locals {
  appengine_bindings = [
    "roles/appengine.deployer",
    "roles/appengine.serviceAdmin",
    "roles/compute.storageAdmin",
    "roles/cloudbuild.builds.builder",
  ]
}

resource "google_project_iam_member" "bindings" {
  project  = module.bootstrap_data.content.project_id
  for_each = toset(local.appengine_bindings)
  role     = each.value
  member   = "serviceAccount:mtrx-hub-dev-3of@appspot.gserviceaccount.com"
}

resource "google_app_engine_application" "appengine_app" {
  project     = module.bootstrap_data.content.project_id
  location_id = substr(var.default_region, 0, length(var.default_region) - 1)
}