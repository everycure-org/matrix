resource "google_storage_bucket_iam_member" "object_user" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.objectCreator"
  member = "group:matrix-all@everycure.org"

  condition {
    title      = "only_edit_raw"
    expression = "resource.name.startsWith('projects/_/buckets/${var.storage_bucket_name}/objects/data/01_RAW')"
  }
}

# Add a new binding for the specific service account
resource "google_storage_bucket_iam_member" "prod_sa_access" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:sa-k8s-node@mtrx-hub-prod-sms.iam.gserviceaccount.com"
}

# ------ Permission for people to read from Storage via SA ------

# Create the service account
resource "google_service_account" "storage_viewer_sa" {
  account_id   = "storage-viewer-sa"
  display_name = "Storage Viewer Service Account"
  description  = "Service account with storage object viewer role"
}

# Create a service account key
resource "google_service_account_key" "storage_viewer_key" {
  service_account_id = google_service_account.storage_viewer_sa.name
}

# Store the key in Secret Manager
resource "google_secret_manager_secret" "storage_viewer_key" {
  secret_id = "storage-viewer-sa-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "storage_viewer_key" {
  secret                = google_secret_manager_secret.storage_viewer_key.id
  secret_data_wo        = base64decode(google_service_account_key.storage_viewer_key.private_key)
  is_secret_data_base64 = false
}

# Grant access to the secret to matrix-all group
resource "google_secret_manager_secret_iam_member" "storage_viewer_key_access" {
  secret_id = google_secret_manager_secret.storage_viewer_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "group:matrix-all@everycure.org"
}

# furthermore grant access to the serviceaccount of wg1/wg2
resource "google_secret_manager_secret_iam_member" "storage_viewer_key_access_wg1" {
  secret_id = google_secret_manager_secret.storage_viewer_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:vertex-ai-workbench-sa@mtrx-wg1-data-dev-nb5.iam.gserviceaccount.com"
}

resource "google_secret_manager_secret_iam_member" "storage_viewer_key_access_wg2" {
  secret_id = google_secret_manager_secret.storage_viewer_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:vertex-ai-workbench-sa@mtrx-wg2-modeling-dev-9yj.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "storage_viewer_iam" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.storage_viewer_sa.email}"
}

resource "google_project_iam_member" "bq_data_viewer" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.storage_viewer_sa.email}"
}
resource "google_project_iam_member" "bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.storage_viewer_sa.email}"
}

resource "google_project_iam_member" "bq_read_session" {
  project = var.project_id
  role    = "roles/bigquery.readSessionUser"
  member  = "serviceAccount:${google_service_account.storage_viewer_sa.email}"
}

# Add a new binding for the matrix-all group to allow object listering
resource "google_storage_bucket_iam_member" "object_lister" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.objectViewer"
  member = "group:matrix-all@everycure.org"
}

# add a new binding for the compute engine default service account for Orchard prod
resource "google_storage_bucket_iam_member" "compute_engine_default_ordchard_prod" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.bucketViewer"
  member = local.orchard_prod_compute_service_account
}

# Add IAM binding for the custom Cloud Build service account from prod
resource "google_storage_bucket_iam_member" "custom_cloud_build_sa_access" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.admin"
  member = "serviceAccount:custom-cloud-build-sa@mtrx-hub-prod-sms.iam.gserviceaccount.com"
}

# add a new binding for the compute engine default service account for Orchard dev
resource "google_storage_bucket_iam_member" "compute_engine_default_ordchard_dev" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.bucketViewer"
  member = local.orchard_dev_compute_service_account
}