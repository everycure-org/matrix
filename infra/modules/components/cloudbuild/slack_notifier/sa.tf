# Create cloud build notifier service account
resource "google_service_account" "notifier" {
  account_id = "${local.base_name}-nfy"
  project    = var.project_id
}

# Give the service account required project permissions
resource "google_project_iam_member" "notifier_project_roles" {
  for_each = toset([
    "roles/storage.objectViewer",
    "roles/iam.serviceAccountTokenCreator",
    "roles/logging.logWriter"
  ])

  project = var.project_id
  role    = each.key
  member  = google_service_account.notifier.member
}

# Give the notifier service account access to the secret
resource "google_secret_manager_secret_iam_member" "notifier_secret_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.slack_webhook_url.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.notifier.member
}

# Look up the pubsub SA
resource "google_project_service_identity" "pubsub" {
  provider = google-beta
  project  = var.project_id
  service  = "pubsub.googleapis.com"
}

# Grant the Pub/Sub SA permission to create auth tokens in your project
resource "google_project_iam_member" "pubsub_project_roles" {
  project = var.project_id
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = google_project_service_identity.pubsub.member
}

# Create a pub/sub invoker service account
resource "google_service_account" "pubsub_invoker" {
  account_id = "${local.base_name}-pbs"
  project    = var.project_id
}

# Give the pub/sub invoker service account the Cloud Run Invoker permission
resource "google_project_iam_member" "pubsub_invoker_roles" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = google_service_account.pubsub_invoker.member
}
