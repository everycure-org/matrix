resource "kubernetes_config_map" "parameters" {
  metadata {
    name      = "matrix-config"
    namespace = "argo-workflows"
    labels = {
      "workflows.argoproj.io/configmap-type" = "Parameter"
    }
  }

  data = {
    GCP_PROJECT_ID = var.project_id
    GCP_BUCKET     = var.bucket_name
  }
}