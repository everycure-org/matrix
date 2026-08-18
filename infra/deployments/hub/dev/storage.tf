resource "google_storage_bucket_iam_member" "object_user" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.objectCreator"
  member = "group:techteam@everycure.org"

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
  member    = "group:techteam@everycure.org"
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
  member = "group:techteam@everycure.org"
}

# add a new binding for the compute engine default service account for Orchard prod
resource "google_storage_bucket_iam_member" "compute_engine_default_ordchard_prod" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.bucketViewer"
  member = local.orchard_prod_compute_service_account
}

# add a new binding for the compute engine default service account for Orchard dev
resource "google_storage_bucket_iam_member" "compute_engine_default_ordchard_dev" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.bucketViewer"
  member = local.orchard_dev_compute_service_account
}

resource "google_project_iam_custom_role" "storage_bucket_lister" {
  project     = var.project_id
  role_id     = "storageBucketLister"
  title       = "Storage Bucket Lister"
  description = "Allows listing storage buckets in the project without granting object access"
  permissions = ["storage.buckets.list"]
}

resource "google_project_iam_member" "matrix_subcontractors_bucket_lister" {
  project = var.project_id
  role    = google_project_iam_custom_role.storage_bucket_lister.id
  member  = "group:matrix-subcontractors@everycure.org"
}

resource "google_storage_bucket_iam_member" "matrix_subcontractors_object_viewer" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.objectViewer"
  member = "group:matrix-subcontractors@everycure.org"
}

resource "google_storage_bucket_iam_member" "matrix_subcontractors_object_creator" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.objectCreator"
  member = "group:matrix-subcontractors@everycure.org"
}

resource "google_storage_bucket_iam_member" "matrix_subcontractors_bucket_reader" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.legacyBucketReader"
  member = "group:matrix-subcontractors@everycure.org"
}

# Temporary scoped access for Alan Hueb (Scripps) to v0.8.2 release data
resource "google_storage_bucket_iam_member" "alan_hueb_release_viewer" {
  bucket = "mtrx-us-central1-hub-dev-storage"
  role   = "roles/storage.objectViewer"
  member = "user:alan@hueb.org"

  condition {
    title      = "only_v0_8_2_release"
    expression = "resource.name.startsWith('projects/_/buckets/mtrx-us-central1-hub-dev-storage/objects/kedro/data/releases/v0.8.2')"
  }
}

# Bucket-level read access for Alan (needed for gsutil/SDK navigation; no condition possible on bucket-level perms)
resource "google_storage_bucket_iam_member" "alan_hueb_bucket_reader" {
  bucket = "mtrx-us-central1-hub-dev-storage"
  role   = "roles/storage.legacyBucketReader"
  member = "user:alan@hueb.org"
}

# Scoped access for Alexei to run `make docker_cloud_build` (gcloud builds submit)
# against the auto-created Cloud Build staging bucket.
resource "google_project_iam_member" "alexei_serviceusage_consumer" {
  project = var.project_id
  role    = "roles/serviceusage.serviceUsageConsumer"
  member  = "user:alexei@everycure.org"
}

resource "google_storage_bucket_iam_member" "alexei_cloudbuild_staging_bucket_admin" {
  bucket = "${var.project_id}_cloudbuild"
  role   = "roles/storage.objectAdmin"
  member = "user:alexei@everycure.org"
}

# needed for bucket-level navigation (e.g. `gcloud builds submit` resolving the bucket before upload)
resource "google_storage_bucket_iam_member" "alexei_cloudbuild_staging_bucket_reader" {
  bucket = "${var.project_id}_cloudbuild"
  role   = "roles/storage.legacyBucketReader"
  member = "user:alexei@everycure.org"
}

# lets alexei create Cloud Build runs directly (equivalent to group grants in modules/components/cloudbuild/sa.tf)
resource "google_project_iam_member" "alexei_cloudbuild_editor" {
  project = var.project_id
  role    = "roles/cloudbuild.builds.editor"
  member  = "user:alexei@everycure.org"
}

# cloudbuild.yaml pins a custom SA; alexei needs to impersonate it to submit builds using it
resource "google_service_account_iam_member" "alexei_cloudbuild_sa_impersonation" {
  service_account_id = "projects/${var.project_id}/serviceAccounts/custom-cloud-build-sa@${var.project_id}.iam.gserviceaccount.com"
  role               = "roles/iam.serviceAccountUser"
  member             = "user:alexei@everycure.org"
}

resource "google_service_account_iam_member" "alexei_cloudbuild_sa_token_creator" {
  service_account_id = "projects/${var.project_id}/serviceAccounts/custom-cloud-build-sa@${var.project_id}.iam.gserviceaccount.com"
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "user:alexei@everycure.org"
}
