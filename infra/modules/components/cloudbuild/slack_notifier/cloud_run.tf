# Cloud Build Notifier
resource "random_id" "cloud_build_notifier_service" {
  # We use a keeper here so we can force cloud run to redeploy on script change.
  keepers = {
    script_hash         = google_storage_bucket_object.cloud_build_notifier_config.md5hash
    slack_template_hash = google_storage_bucket_object.slack_template.md5hash
  }

  byte_length = 4
}

resource "google_cloud_run_service" "cloud_build_notifier" {
  provider = google-beta
  name     = "${local.base_name}-${random_id.cloud_build_notifier_service.hex}"
  location = var.region
  project  = var.project_id

  template {
    spec {
      service_account_name = google_service_account.notifier.email

      containers {
        image = var.cloud_build_notifier_image

        env {
          name  = "CONFIG_PATH"
          value = "${google_storage_bucket.cloud_build_notifier.url}/${local.base_name}-config.yaml"
        }

        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }
      }
    }
  }

  autogenerate_revision_name = true

  lifecycle {
    # Ignored because Cloud Run may add annotations outside of this config
    ignore_changes = [
      metadata.0.annotations,
    ]
  }

  depends_on = [
    google_secret_manager_secret_iam_member.notifier_secret_accessor
  ]
}