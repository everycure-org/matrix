resource "google_storage_bucket_iam_member" "object_user" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.objectCreator"
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

resource "google_project_iam_member" "storage_viewer_iam" {
  project = module.bootstrap_data.content.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.storage_viewer_sa.email}"
}
resource "google_project_iam_member" "bq_data_viewer" {
  project = module.bootstrap_data.content.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.storage_viewer_sa.email}"
}
resource "google_project_iam_member" "bq_job_user" {
  project = module.bootstrap_data.content.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.storage_viewer_sa.email}"
}

resource "google_project_iam_member" "bq_read_session" {
  project = module.bootstrap_data.content.project_id
  role    = "roles/bigquery.readSessionUser"
  member  = "serviceAccount:${google_service_account.storage_viewer_sa.email}"
}
