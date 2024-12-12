resource "kubernetes_config_map" "parameters" {
  metadata {
    name      = "moa-config"
    namespace = "moa"
  }

  data = {
    GCP_PROJECT_ID = var.project_id
    GCP_BUCKET     = var.bucket_name
    APP_IMAGE      = "us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/moa_visualizer:"
  }
}