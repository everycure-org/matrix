data "google_compute_default_service_account" "default" {
  project = var.project_id
}

data "google_project" "project" {
  project_id = var.project_id
}

// P4SA stands for Project for Service Account or more precisely, **Per-Project Per-Service Service Account**.
// P4SA is a special Google-managed service account that a Google Cloud service (like Cloud Build, Cloud Functions, or Vertex AI) uses within your project to perform actions on your behalf.
data "google_iam_policy" "p4sa-secretAccessor" {
  binding {
    role    = "roles/secretmanager.secretAccessor"
    members = [data.google_compute_default_service_account.default.member]
  }
}