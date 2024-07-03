locals {
  k8s_sa_roles = [
    "roles/container.defaultNodeServiceAccount",
    "roles/storage.objectAdmin",
    "roles/bigquery.dataEditor",
    "roles/bigquery.user",
    "roles/bigquery.jobUser",
    "roles/aiplatform.user",
    "roles/artifactregistry.writer",
    "roles/artifactregistry.reader",
  ]
}

resource "google_project_iam_member" "k8s_sa_bindings" {
  for_each = toset(local.k8s_sa_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${module.gke.service_account}"

}