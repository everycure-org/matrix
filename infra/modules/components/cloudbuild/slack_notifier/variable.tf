variable "project_id" {
  description = "The ID of the GCP project where the Cloud Build notifier will be deployed."
  type        = string
}

variable "slack_webhook_url" {
  description = "The Slack webhook URL to send notifications to."
  type        = string
  sensitive   = true
}

# See: https://cloud.google.com/build/docs/configuring-notifications/configure-slack#using_cel_to_filter_build_events
variable "cloud_build_event_filter" {
  description = "The CEL filter to apply to incoming Cloud Build events."
  type        = string
  default     = "'BRANCH_NAME' in build.substitutions && build.status in [Build.Status.FAILURE, Build.Status.TIMEOUT]"
}

variable "cloud_build_notifier_image" {
  description = "The image to use for the notifier."
  type        = string
  default     = "us-east1-docker.pkg.dev/gcb-release/cloud-build-notifiers/slack:latest"
}

variable "region" {
  description = "The region where the Cloud Build resources will be created."
  type        = string
  default     = "us-central1"
}