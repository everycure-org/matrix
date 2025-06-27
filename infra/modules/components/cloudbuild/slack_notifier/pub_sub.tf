# Create the cloud-builds topic to receive build update messages for your notifier
resource "google_pubsub_topic" "cloud_builds" {
  project = var.project_id
  name    = "cloud-builds"
}

resource "google_pubsub_subscription" "cloud_builds" {
  name    = local.base_name
  topic   = google_pubsub_topic.cloud_builds.name
  project = var.project_id

  push_config {
    push_endpoint = google_cloud_run_service.cloud_build_notifier.status[0].url

    oidc_token {
      service_account_email = google_service_account.pubsub_invoker.email
    }
  }
}