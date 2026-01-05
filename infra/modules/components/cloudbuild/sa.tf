# Allow engineering group to impersonate the Cloud Build service account
resource "google_service_account_iam_member" "engineering_cloudbuild_impersonation" {
  service_account_id = google_service_account.cloudbuild_sa.id
  role               = "roles/iam.serviceAccountUser"
  member             = "group:engineering@everycure.org"
}

# Allow engineering group to create tokens for the Cloud Build service account
resource "google_service_account_iam_member" "engineering_cloudbuild_token_creator" {
  service_account_id = google_service_account.cloudbuild_sa.id
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "group:engineering@everycure.org"
}
