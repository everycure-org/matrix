# Create OAuth client
resource "google_iap_client" "matrix_cli_client" {
  display_name = "Matrix CLI OAuth Client"
  brand        = "Every Cure Dev Platform"
}

# Create a secret in Secret Manager for the OAuth client secret
resource "google_secret_manager_secret" "oauth_client_secret" {
  secret_id = "matrix-cli-oauth-client-secret"

  replication {
    auto {}
  }
}

# Store the OAuth client secret in Secret Manager
resource "google_secret_manager_secret_version" "oauth_client_secret_version" {
  secret      = google_secret_manager_secret.oauth_client_secret.id
  secret_data = google_iap_client.matrix_cli_client.secret
}

# IAM binding to allow necessary service accounts to access the secret
resource "google_secret_manager_secret_iam_member" "secret_access" {
  secret_id = google_secret_manager_secret.oauth_client_secret.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "group:matrix-all@everycure.org"
}