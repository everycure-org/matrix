data "google_compute_default_service_account" "default" {
  project = var.project_id
}

data "google_project" "project" {
  project_id = var.project_id
}

data "google_iam_policy" "p4sa-secretAccessor" {
  binding {
    role    = "roles/secretmanager.secretAccessor"
    members = [data.google_compute_default_service_account.default.member]
  }
}