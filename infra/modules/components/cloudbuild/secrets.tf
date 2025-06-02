resource "google_secret_manager_secret" "github_token" {
  secret_id = "github-token-cloudbuild"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "github_token_version" {
  secret         = google_secret_manager_secret.github_token.id
  secret_data_wo = var.github_repo_token
}

resource "google_secret_manager_secret" "gitcrypt_key" {
  secret_id = "gitcrypt-key-cloudbuild"
  project   = var.project_id
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "gitcrypt_key_version" {
  secret         = google_secret_manager_secret.gitcrypt_key.id
  secret_data_wo = var.gitcrypt_key
  // The gitcrypt key is base64 encoded, and we store it as a base64 string in the secret manager.
  // This is a workaround for the fact that the secret manager does not support binary data.
  is_secret_data_base64 = false
}

resource "google_secret_manager_secret" "slack_webhook_url" {
  secret_id = "github-token-cloudbuild"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "slack_webhook_url_version" {
  secret         = google_secret_manager_secret.slack_webhook_url.id
  secret_data_wo = var.slack_webhook_url
}
