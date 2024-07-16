# **************** GCP side prep **********************
data "google_project" "project" {
  project_id = var.project_id
}
locals {
  es_sa = "gcp-sm-reader"
  es_ns = "external-secrets"
  # es_iam_member = "principal://iam.googleapis.com/projects/${data.google_project.project.number}/locations/global/workloadIdentityPools/${var.project_id}.svc.id.goog/subject/ns/${local.es_ns}/sa/${local.es_sa}"
}

resource "kubernetes_service_account" "external-secrets-sa" {
  metadata {
    name = local.es_sa
  }
}

resource "google_service_account" "external_secrets_sa" {
  project      = var.project_id
  account_id   = "sa-gke-external-secrets"
  display_name = "sa-gke-external-secrets"
  description  = "Used for external secrets in k8s to access the secrets of GCP secrets manager"
}

resource "google_service_account_key" "external_secrets_sa_key" {
  service_account_id = google_service_account.external_secrets_sa.name
}

resource "google_project_iam_member" "external_secrets_gke_sa" {
  project = var.project_id
  role    = "roles/secretmanager.viewer"
  member  = google_service_account.external_secrets_sa.member
}
resource "google_project_iam_member" "external_secrets_gke_sa_access" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = google_service_account.external_secrets_sa.member
}

# **************** K8S side prep **********************
resource "kubernetes_namespace" "external_secrets_ns" {
  depends_on = [module.gke]
  metadata {
    name = "external-secrets"
  }
}

resource "kubernetes_secret" "external_secrets_sa" {
  metadata {
    name      = "gcp-sa-key"
    namespace = kubernetes_namespace.external_secrets_ns.metadata[0].name
    labels = {
      type = "gcpsm"
    }
  }
  data = {
    sa_key = base64decode(google_service_account_key.external_secrets_sa_key.private_key)
  }
  type = "Opaque"
}