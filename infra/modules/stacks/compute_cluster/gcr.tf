module "image_repo" {
  source = "../../artifact_registry"

  project_id    = var.project_id
  location      = var.default_region
  repository_id = "matrix-images"
  description   = "images for matrix project"
  format        = "DOCKER"
}