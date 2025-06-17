module "slack_notifier" {
  source            = "./slack_notifier"
  project_id        = var.project_id
  slack_webhook_url = var.slack_webhook_url
}