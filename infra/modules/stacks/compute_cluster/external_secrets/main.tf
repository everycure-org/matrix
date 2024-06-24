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