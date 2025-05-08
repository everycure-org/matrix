resource "google_cloudbuildv2_connection" "github" {
  name     = "github-connection"
  location = "global"

  github_config {
    app_installation_id = var.github_app_installation_id
    authorizer_credential {
      oauth_token_secret_version = google_secret_manager_secret_version.github_token_version.id
    }
  }
}
