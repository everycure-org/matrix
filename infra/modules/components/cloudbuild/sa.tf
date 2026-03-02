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

# Allow data-science group to impersonate the Cloud Build service account
resource "google_service_account_iam_member" "data-science_cloudbuild_impersonation" {
  service_account_id = google_service_account.cloudbuild_sa.id
  role               = "roles/iam.serviceAccountUser"
  member             = "group:data-science@everycure.org"
}

# Allow data-science group to create tokens for the Cloud Build service account
resource "google_service_account_iam_member" "data-science_cloudbuild_token_creator" {
  service_account_id = google_service_account.cloudbuild_sa.id
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "group:data-science@everycure.org"
}

# Allow matrix-subcontractors group to impersonate the Cloud Build service account
resource "google_service_account_iam_member" "matrix_subcontractors_cloudbuild_impersonation" {
  service_account_id = google_service_account.cloudbuild_sa.id
  role               = "roles/iam.serviceAccountUser"
  member             = "group:matrix-subcontractors@everycure.org"
}

# Allow matrix-subcontractors group to create tokens for the Cloud Build service account
resource "google_service_account_iam_member" "matrix_subcontractors_cloudbuild_token_creator" {
  service_account_id = google_service_account.cloudbuild_sa.id
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "group:matrix-subcontractors@everycure.org"
}
