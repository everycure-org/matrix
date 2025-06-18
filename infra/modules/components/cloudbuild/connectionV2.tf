// This module creates a Google Cloud Build connection to a GitHub repository using the Cloud Build API v2.
resource "google_cloudbuildv2_connection" "github_connection" {
  name     = "github-connection"
  location = var.location
  project  = var.project_id

  github_config {
    app_installation_id = var.github_app_installation_id
    authorizer_credential {
      oauth_token_secret_version = google_secret_manager_secret_version.github_token_version.id
    }
  }
  depends_on = [google_secret_manager_secret_iam_policy.policy]
}

resource "google_cloudbuildv2_repository" "matrix_repo" {
  project           = var.project_id
  location          = var.location
  name              = var.github_repo_name
  parent_connection = google_cloudbuildv2_connection.github_connection.name
  remote_uri        = "https://github.com/${var.github_repo_owner}/${var.github_repo_name}.git"
}