resource "google_iap_brand" "matrix_google_iap_brand" {
  support_email     = "matrix-all@everycure.org" # Using the existing group email
  application_title = "Matrix Hub Dev Application"
  project           = var.project_id
}

# Create OAuth client for desktop application
resource "google_iap_client" "desktop_client" {
  display_name = "Matrix CLI OAuth Client"
  brand        = google_iap_brand.matrix_google_iap_brand.name
}

# Create a secret in Secret Manager for the OAuth client secret
resource "google_secret_manager_secret" "oauth_client_secret" {
  secret_id = "desktop-app-oauth-client-secret"

  replication {
    auto {}
  }
}

# Store the OAuth client secret in Secret Manager
resource "google_secret_manager_secret_version" "oauth_client_secret_version" {
  secret      = google_secret_manager_secret.oauth_client_secret.id
  secret_data = google_iap_client.desktop_client.secret
}

# IAM binding to allow necessary service accounts to access the secret
resource "google_secret_manager_secret_iam_member" "secret_access" {
  secret_id = google_secret_manager_secret.oauth_client_secret.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "group:matrix-all@everycure.org"
}