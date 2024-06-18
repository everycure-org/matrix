# documentation page. Only hosted in dev for now

# IAM SA for Cloud Build
# IAM SA for Cloud Run
# Cloudbuild default SA config set to above SA
# Cloud Build Trigger linked to GH
# Cloud Run deployment
# Artifact registry  

# needed for whatever reason 
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

#resource "google_iap_tunnel_dest_group" "dest_group" {
#  region = "us-central1"
#  group_name = "internal-"
#  cidrs = [
#    "10.1.0.0/16",
#    "192.168.10.0/24",
#  ]
#}

import {
  id = "mtrx-hub-dev-3of"
  to = google_app_engine_application.appengine_app
}

resource "google_app_engine_application" "appengine_app" {
  project        = module.bootstrap_data.content.project_id
  location_id    = substr(var.default_region, 0, length(var.default_region)-1)
#   auth_domain    = var.auth_domain
#   database_type  = var.database_type
#   serving_status = var.serving_status
  # dynamic "feature_settings" {
  #   for_each = var.feature_settings[*]
  #   content {
  #     split_health_checks = feature_settings.value.split_health_checks
  #   }
  # }
  # dynamic "iap" {
  #   for_each = var.iap[*]
  #   content {
  #     oauth2_client_id     = iap.value.oauth2_client_id
  #     oauth2_client_secret = iap.value.oauth2_client_secret
  #   }
  # }
}