resource "google_service_account" "cloudbuild_sa" {
  account_id   = "custom-cloud-build-sa"
  display_name = "Custom Cloud Build Service Account"
  project      = var.project_id
}

resource "google_project_iam_member" "cloudbuild_builder" {
  project = var.project_id
  role    = "roles/owner"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}