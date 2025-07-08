resource "google_secret_manager_secret" "slack_webhook_url" {
  secret_id = "slack-webhook-url-cloudbuild"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "slack_webhook_url_version" {
  secret         = google_secret_manager_secret.slack_webhook_url.id
  secret_data_wo = var.slack_webhook_url
}