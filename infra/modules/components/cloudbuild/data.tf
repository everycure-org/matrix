data "google_compute_default_service_account" "default" {
  project = var.project_id
}

data "google_project" "project" {
  project_id = var.project_id
}

// P4SA stands for Project for Service Account or more precisely, **Per-Project Per-Service Service Account**.
// P4SA is a special Google-managed service account that a Google Cloud service (like Cloud Build, Cloud Functions, or Vertex AI) uses within your project to perform actions on your behalf.
data "google_iam_policy" "p4sa-secretAccessor" {
  binding {
    role = "roles/secretmanager.secretAccessor"
    // Sadly, the P4SA for Cloud Build is not documented in the official documentation and there is no way to get it programmatically. See: https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/cloudbuildv2_repository#:~:text=for%20the%20project%20that%20contains%20the%20connection.-,members%20%3D%20%5B%22serviceAccount%3Aservice%2D123456789%40gcp%2Dsa%2Dcloudbuild.iam.gserviceaccount.com%22%5D,-%7D%0A%7D%0A%0Aresource%20%22google_secret_manager_secret_iam_policy%22%20%22policy%2Dpk
    members = ["serviceAccount:service-${data.google_project.project.number}@gcp-sa-cloudbuild.iam.gserviceaccount.com"]

  }
}