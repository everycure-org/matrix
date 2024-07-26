resource "google_storage_bucket_iam_member" "object_user" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.objectUser"
  member = "group:matrix-all@everycure.org"

  condition {
    title      = "only_edit_raw"
    expression = "resource.name.startsWith('projects/_/buckets/${var.storage_bucket_name}/objects/data/01_RAW')"
  }
}


# ------ Permission for people to read from Storage via SA ------

# Create the service account
resource "google_service_account" "storage_viewer_sa" {
  account_id   = "storage-viewer-sa"
  display_name = "Storage Viewer Service Account"
  description  = "Service account with storage object viewer role"
}

# Assign the storage.objectViewer role to the service account
resource "google_project_iam_member" "storage_viewer_iam" {
  project = module.bootstrap_data.content.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.storage_viewer_sa.email}"
}

# Create a custom IAM role that allows creating service account keys
resource "google_project_iam_custom_role" "sa_key_creator" {
  role_id     = "serviceAccountKeyCreator"
  title       = "Service Account Key Creator"
  description = "Custom role to create service account keys"
  permissions = ["iam.serviceAccountKeys.create"]
}

# Assign the custom role to the matrix-all group for the specific service account
resource "google_service_account_iam_member" "matrix_all_sa_key_creator" {
  service_account_id = google_service_account.storage_viewer_sa.name
  role               = google_project_iam_custom_role.sa_key_creator.id
  member             = "group:matrix-all@everycure.org"
}