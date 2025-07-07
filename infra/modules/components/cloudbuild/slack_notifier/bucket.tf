# Create bucket
resource "random_id" "cloud_build_notifier" {
  byte_length = 4
}

resource "google_storage_bucket" "cloud_build_notifier" {
  project                     = var.project_id
  name                        = "${local.base_name}-${random_id.cloud_build_notifier.hex}"
  force_destroy               = true
  location                    = var.region
  uniform_bucket_level_access = true
}

locals {
  slack_template_json = file("${path.module}/slack.json")
}

resource "google_storage_bucket_object" "slack_template" {
  name          = "${local.base_name}-slack.json"
  bucket        = google_storage_bucket.cloud_build_notifier.name
  content       = local.slack_template_json
  storage_class = "NEARLINE"
}

resource "google_storage_bucket_object" "cloud_build_notifier_config" {
  name   = "${local.base_name}-config.yaml"
  bucket = google_storage_bucket.cloud_build_notifier.name

  storage_class = "NEARLINE"

  content = jsonencode({
    apiVersion = "cloud-build-notifiers/v1"
    kind       = "SlackNotifier"
    metadata = {
      name = local.base_name
    }
    spec = {
      notification = {
        filter = var.cloud_build_event_filter
        delivery = {
          webhookUrl = {
            secretRef = "webhook-url"
          }
        }
        template = {
          type = "golang"
          uri  = "${google_storage_bucket.cloud_build_notifier.url}/${local.base_name}-slack.json"
        }
      }
      secrets = [
        {
          name  = "webhook-url"
          value = google_secret_manager_secret_version.slack_webhook_url_version.name
        }
      ]
    }
  })
}
