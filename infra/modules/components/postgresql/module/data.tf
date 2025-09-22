# Get the secret version from Google Secret Manager
data "google_secret_manager_secret_version" "secret" {
  secret  = var.secret_name
  project = var.project_id
}