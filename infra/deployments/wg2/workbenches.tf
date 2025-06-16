locals {
  users = yamldecode(file("workbenches.yaml"))
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
  secret         = google_secret_manager_secret.github_token.id
  secret_data_wo = var.github_token
}


# expanding permissions for SA

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