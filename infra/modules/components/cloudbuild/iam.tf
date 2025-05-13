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

resource "google_secret_manager_secret_iam_policy" "policy" {
  secret_id   = google_secret_manager_secret.github_token.secret_id
  project     = var.project_id
  policy_data = data.google_iam_policy.p4sa-secretAccessor.policy_data
}