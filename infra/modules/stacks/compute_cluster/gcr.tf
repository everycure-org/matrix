resource "google_artifact_registry_repository" "image_repo" {
  location      = var.default_region
  project       = var.project_id
  repository_id = "matrix-images"
  description   = "images for matrix project"
  format        = "DOCKER"
}