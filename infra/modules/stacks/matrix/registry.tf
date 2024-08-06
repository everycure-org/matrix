resource "google_artifact_registry_repository" "mflow-repo" {
  location      = var.default_region
  repository_id = "mlflow"
  description   = "MLFlow docker repository"
  format        = "DOCKER"
}