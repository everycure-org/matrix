# BigQuery
resource "google_project_iam_audit_config" "bigquery_data_access" {
  project = var.project_id
  service = "bigquery.googleapis.com"

  audit_log_config {
    log_type = "DATA_READ"
  }

  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

# Cloud Storage
resource "google_project_iam_audit_config" "gcs_data_access" {
  project = var.project_id
  service = "storage.googleapis.com"

  audit_log_config {
    log_type = "DATA_READ"
  }

  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

# Artifact Registry
resource "google_project_iam_audit_config" "artifactregistry_data_access" {
  project = var.project_id
  service = "artifactregistry.googleapis.com"

  audit_log_config {
    log_type = "DATA_READ"
  }

  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

# Kubernetes Engine API
resource "google_project_iam_audit_config" "gke_data_access" {
  project = var.project_id
  service = "container.googleapis.com"

  audit_log_config {
    log_type = "DATA_READ"
  }

  audit_log_config {
    log_type = "DATA_WRITE"
  }
}