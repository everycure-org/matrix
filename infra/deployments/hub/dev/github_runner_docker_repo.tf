module "github_runner_image" {
  source = "../../../modules/artifact_registry"

  project_id                 = var.project_id
  location                   = var.default_region
  repository_id              = "github-runner-images"
  description                = "images for GitHub runner"
  format                     = "DOCKER"
  keep_number_of_last_images = 1
}