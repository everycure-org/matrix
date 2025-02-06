locals {
  users = {
    "alexei" = "alexei@everycure.org"
    "pascal" = "pascal@everycure.org"
  }
}

module "workbenches" {
  source   = "./modules/workbench"
  for_each = local.users

  username = each.key
  email    = each.value

  network = var.shared_network_name
  subnet  = var.shared_subnetwork_name

  service_account = "vertex-ai-workbench-sa@${var.project_id}.iam.gserviceaccount.com"
  project_id      = var.project_id

  # Optional: Override defaults if needed
  # machine_type = "e2-standard-8"
  # boot_disk_size_gb = 200
  # data_disk_size_gb = 150
  post_startup_script = local.gcs_path
}

locals {
  gcs_path = "gs://${var.storage_bucket_name}/post_startup_script.sh"
}

resource "google_storage_bucket_object" "post_startup_script" {
  name   = "post_startup_script.sh"
  bucket = var.storage_bucket_name
  source = "./setup_vertex_workbench.sh"
}

# secret for github that allows cloning matrix repo only
resource "google_secret_manager_secret" "github_token" {
  secret_id = "github-token"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "github_token" {
  secret      = google_secret_manager_secret.github_token.id
  secret_data = var.github_token
}



# inspiration from 
# https://github.com/GoogleCloudPlatform/terraform-google-cloud-functions/blob/v0.6.0/examples/cloud_function2_pubsub_trigger/main.tf
module "pubsub" {
  source  = "terraform-google-modules/pubsub/google"
  version = "~> 6.0"

  topic      = "workbench-idle-alert"
  project_id = var.project_id
}

# Create notification channel for PubSub

# Create Cloud Function to log PubSub events

data "archive_file" "function_zip" {
  type        = "zip"
  source_dir  = "${path.module}/functions/workbench_event_handler"
  excludes    = ["build"]
  output_path = "${path.module}/functions/workbench_event_handler/build/function.zip"
}

# ========================================================
# Monitoring alerts
# ========================================================

resource "google_storage_bucket_object" "function_code" {
  name   = "functions/workbench_event_handler/function-${data.archive_file.function_zip.output_md5}.zip"
  bucket = var.storage_bucket_name
  source = data.archive_file.function_zip.output_path
}

resource "google_monitoring_notification_channel" "pubsub" {
  display_name = "Workbench Idle Alert Channel"
  type         = "pubsub"
  labels = {
    topic = module.pubsub.id
  }
}

# Create alert policy
resource "google_monitoring_alert_policy" "alert_policy" {
  display_name = "wg2-idle-instance-alert-pubsub"
  documentation {
    content   = "If you left one running go here \nhttps://console.cloud.google.com/vertex-ai/workbench/instances?inv=1&invt=AbnGiQ&project=mtrx-wg2-modeling-dev-9yj\n\nto turn it off please. "
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "VM Instance - CPU utilization"
    condition_threshold {
      filter          = "resource.type = \"gce_instance\" AND metric.type = \"compute.googleapis.com/instance/cpu/utilization\""
      duration        = "0s"
      threshold_value = 0.1
      comparison      = "COMPARISON_LT"
      trigger {
        count = 1
      }
      aggregations {
        alignment_period   = "10800s"
        per_series_aligner = "ALIGN_MAX"
      }
    }
  }

  combiner = "OR"
  enabled  = true

  notification_channels = [
    google_monitoring_notification_channel.pubsub.name
  ]
}

# ========================================================
# Cloud Function to handle PubSub events
# ========================================================

# expanding permissinos for SA

locals {
  permissions = [
    "roles/logging.logWriter",
    "roles/storage.objectViewer",
    "roles/artifactregistry.writer",
    "roles/run.invoker",
    "roles/reader",
    "roles/secretmanager.secretAccessor",
    "roles/notebooks.runner",
    "roles/notebooks.viewer"
  ]
}
resource "google_project_iam_member" "workbench_sa_permissions" {
  for_each = toset(local.permissions)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:vertex-ai-workbench-sa@${var.project_id}.iam.gserviceaccount.com"
}


module "workbench_alert_function" {
  source = "GoogleCloudPlatform/cloud-functions/google"
  depends_on = [
    google_project_iam_member.workbench_sa_permissions
  ]
  version = "0.6.0"

  entrypoint        = "pubsub_handler"
  function_location = var.location

  project_id            = var.project_id
  function_name         = "workbench-alert-logger"
  build_service_account = "projects/${var.project_id}/serviceAccounts/vertex-ai-workbench-sa@${var.project_id}.iam.gserviceaccount.com"

  runtime = "python310"

  event_trigger = {
    trigger_region        = var.location
    event_type            = "google.cloud.pubsub.topic.v1.messagePublished"
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = "vertex-ai-workbench-sa@${var.project_id}.iam.gserviceaccount.com"
    pubsub_topic          = module.pubsub.id
  }

  storage_source = {
    bucket = var.storage_bucket_name
    object = google_storage_bucket_object.function_code.name
  }

  service_config = {
    max_instance_count  = 1
    available_memory_mb = 256
    timeout_seconds     = 540
  }

  labels = {
    purpose = "workbench-monitoring"
  }
}
