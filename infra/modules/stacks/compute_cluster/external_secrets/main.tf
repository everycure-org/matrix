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

module "service_accounts" {
  source     = "terraform-google-modules/service-accounts/google"
  version    = "~> 4.0"
  project_id = var.project_id
  prefix     = "sa-"
  names      = ["gke-external-secrets"]
  project_roles = [
    # TODO: restrict roles further here
    "${var.project_id}=>roles/owner",
  ]
}