locals {
  secrets_output = {
    for k, _ in var.k8s_secrets : k => google_secret_manager_secret.secrets[k].name
  }
  # wrapping as nonsensitive to be able to apply for_each. This requires careful treatment of the secrets to avoid them being leaked as for_each key
  secret_keys = nonsensitive(toset(keys(var.k8s_secrets)))
}

resource "google_secret_manager_secret" "secrets" {
  for_each  = local.secret_keys
  secret_id = each.key
  project   = var.project_id

  replication {
    auto {}
  }
  labels = {
    purpose = "k8s"
    cluster = var.name
  }
}

resource "google_secret_manager_secret_version" "secret_values" {
  for_each    = local.secret_keys
  secret      = google_secret_manager_secret.secrets[each.key].id
  secret_data = var.k8s_secrets[each.key]
}

# secrets manager bindings
data "google_project" "project" {
  project_id = var.project_id
}
locals {
  es_sa         = "default"
  es_ns         = "external-secrets"
  es_iam_member = "principal://iam.googleapis.com/projects/${data.google_project.project.number}/locations/global/workloadIdentityPools/${var.project_id}.svc.id.goog/subject/ns/${local.es_ns}/sa/${local.es_sa}"
}

resource "google_project_iam_member" "external_secrets_gke_sa" {
  project = var.project_id
  role    = "roles/secretmanager.viewer"
  member  = local.es_iam_member
}

resource "google_project_iam_member" "external_secrets_gke_sa_access" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = local.es_iam_member
}