resource "helm_release" "external_secrets" {
  name             = "external-secrets"
  namespace        = var.namespace
  create_namespace = "true"
  repository       = "https://charts.external-secrets.io"
  chart            = "external-secrets"
  version          = "0.9.19"
  set {
    name  = "installCRDs"
    value = "true"
  }
  values = [
    yamlencode({
      "serviceAccount" : {
        "annotations" : var.sa_annotations
      }
    })
  ]
}

resource "kubernetes_manifest" "gcp_store" {
  manifest = {
    "apiVersion" = "external-secrets.io/v1beta1"
    "kind" = "ClusterSecretStore"
    "metadata" = {
      "name" = "gcp-store"
    }
    "spec" = {
      "provider" = {
        "gcpsm" = {
          "projectID" = var.project_id
          "auth" = {
            "workloadIdentity" = {
              "clusterLocation" = var.default_region
              "clusterName" = var.cluster_name
              "serviceAccountRef" = {
                "name" = "default"
                "namespace" = var.namespace
              }
            }
          }
        }
      }
    }
  }
}
